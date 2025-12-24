#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/mul.cuh"
#include "rv32im/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32MultiplicationCoreRecord = MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32MultiplicationCore = MultiplicationCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv32MultiplicationCoreCols = MultiplicationCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv32MultiplicationCols {
    Rv32MultAdapterCols<T> adapter;
    Rv32MultiplicationCoreCols<T> core;
};

struct Rv32MultiplicationRecord {
    Rv32MultAdapterRecord adapter;
    Rv32MultiplicationCoreRecord core;
};

__global__ void mul_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv32MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
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
        is_apc ? sizeof(Rv32MultiplicationCols<uint8_t>) * (idx % calls_per_apc_row) : 0,
        subs,
        is_apc
    );

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv32MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row_new(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv32MultiplicationCore core(range_tuple_checker);
        core.fill_trace_row_new(row.slice_from(COL_INDEX(Rv32MultiplicationCols, core)), rec.core);
    } else {
        if (!is_apc) {
            row.fill_zero(0, sizeof(Rv32MultiplicationCols<uint8_t>));
        } else if (idx < height * calls_per_apc_row) {
            row.fill_zero(0, d_opt_widths[idx % calls_per_apc_row]);
        }
    }
}

extern "C" int _mul_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
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
    bool is_apc = apc_width != 0;
    if (!is_apc) {
        assert(width == sizeof(Rv32MultiplicationCols<uint8_t>));
    }
    size_t threads = is_apc ? (apc_height * calls_per_apc_row) : height;
    auto [grid, block] = kernel_launch_params(threads, 512);

    mul_tracegen<<<grid, block>>>(
        d_trace,
        is_apc ? apc_height : height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits,
        subs,
        d_opt_widths,
        d_post_opt_offsets,
        apc_width,
        calls_per_apc_row
    );
    return CHECK_KERNEL();
}
