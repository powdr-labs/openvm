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
    Fp *trace,
    size_t height,
    DeviceBufferConstView<LessThanRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    ApcParams apc
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row = RowSlice::create_apc_aware(
        trace, height, idx, sizeof(LessThanCols<uint8_t>), apc
    );

    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv32BaseAluAdapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = Rv32LessThanCore(BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThanCols, core)), record.core);
    } else {
        FILL_DUMMY_ROW_APC(row, sizeof(LessThanCols<uint8_t>), idx, height, apc);
    }
}

extern "C" int _rv32_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    ApcParams apc
) {
    // We require the height to be a power of two for the tracegen to work
    assert((height & (height - 1)) == 0);
    assert((apc.height & (apc.height - 1)) == 0);
    assert(height >= d_records.len());
    if (!apc.is_apc()) assert(width == sizeof(LessThanCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(apc.thread_count(height));
    rv32_less_than_tracegen<<<grid, block>>>(
        d_trace,
        apc.effective_height(height),
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits,
        apc
    );
    return CHECK_KERNEL();
}