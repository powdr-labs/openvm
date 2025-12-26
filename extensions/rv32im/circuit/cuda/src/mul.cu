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
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv32MultiplicationRecord> records,
    uint32_t *range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    ApcParams apc
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row = RowSlice::create_apc_aware(
        trace, height, idx, sizeof(Rv32MultiplicationCols<uint8_t>), apc
    );

    if (idx < records.len()) {
        auto const &rec = records[idx];

        Rv32MultAdapter adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv32MultiplicationCore core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32MultiplicationCols, core)), rec.core);
    } else {
        FILL_DUMMY_ROW_APC(row, sizeof(Rv32MultiplicationCols<uint8_t>), idx, height, apc);
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
    ApcParams apc
) {
    assert((height & (height - 1)) == 0);
    assert((apc.height & (apc.height - 1)) == 0);
    assert(height >= d_records.len());
    if (!apc.is_apc()) assert(width == sizeof(Rv32MultiplicationCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(apc.thread_count(height), 512);
    mul_tracegen<<<grid, block>>>(
        d_trace,
        apc.effective_height(height),
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits,
        apc
    );
    return CHECK_KERNEL();
}
