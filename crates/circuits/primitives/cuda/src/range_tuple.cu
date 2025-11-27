#include <cstdio>

#include "fp.h"
#include "launcher.cuh"
#include "primitives/row_print_buffer.cuh"

__global__ void range_tuple_checker_tracegen(
    const uint32_t *count,
    const uint32_t *cpu_count,
    Fp *trace,
    size_t num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // RowPrintBuffer buffer;
    // buffer.reset();
    // buffer.append_literal("num_bins=");
    // buffer.append_uint(num_bins);
    // buffer.append_literal("\n");
    // buffer.flush();
    if (idx < num_bins) {
        // printf("idx=%u\n", idx);
        trace[idx] = Fp(count[idx] + (cpu_count ? cpu_count[idx] : 0));
    }
}

extern "C" int _range_tuple_checker_tracegen(
    const uint32_t *d_count,
    const uint32_t *d_cpu_count,
    Fp *d_trace,
    size_t num_bins
) {
    auto [grid, block] = kernel_launch_params(num_bins);
    range_tuple_checker_tracegen<<<grid, block>>>(d_count, d_cpu_count, d_trace, num_bins);
    return CHECK_KERNEL();
}
