#pragma once

#include "fp.h"
#include "primitives/row_print_buffer.cuh"
#include <cstddef>
#include <cstdint>
#include <type_traits>

/// APC (Automatic Proof Composition) parameters passed from Rust to CUDA.
/// Bundles all APC-related data into a single struct for cleaner interfaces.
struct ApcParams {
    uint32_t *subs;
    uint32_t *opt_widths;
    uint32_t *post_opt_offsets;
    size_t height;
    size_t width;           // 0 = non-APC
    uint32_t calls_per_row; // 1 = non-APC

    __device__ __host__ bool is_apc() const { return width != 0; }

    __host__ size_t thread_count(size_t non_apc_height) const {
        return is_apc() ? (height * calls_per_row) : non_apc_height;
    }

    __host__ size_t effective_height(size_t non_apc_height) const {
        return is_apc() ? height : non_apc_height;
    }
};

__device__ __forceinline__ size_t number_of_gaps_in(const uint32_t *sub, size_t start, size_t len);

/// A RowSlice is a contiguous section of a row in col-based trace.
/// Supports both direct trace access (non-APC) and APC-aware access with column substitution.
struct RowSlice {
    Fp *ptr;
    size_t stride;
    size_t optimized_offset;
    size_t dummy_offset;
    uint32_t *subs;
    bool is_apc;

    // Full constructor for APC-aware access
    __device__ RowSlice(Fp *ptr, size_t stride, size_t optimized_offset, size_t dummy_offset, uint32_t *subs, bool is_apc) : ptr(ptr), stride(stride), optimized_offset(optimized_offset), dummy_offset(dummy_offset), subs(subs), is_apc(is_apc) {}

    // Simple constructor for backward compatibility (non-APC mode)
    __device__ RowSlice(Fp *ptr, size_t stride) : ptr(ptr), stride(stride), optimized_offset(0), dummy_offset(0), subs(nullptr), is_apc(false) {}

    /// Create a RowSlice with APC-aware setup.
    /// When apc_width != 0, computes the correct pointer offset and metadata for APC mode.
    /// When apc_width == 0, creates a simple non-APC RowSlice.
    /// @param d_trace Base pointer to the trace buffer
    /// @param height Trace height (stride between columns)
    /// @param idx Thread index (blockIdx.x * blockDim.x + threadIdx.x)
    /// @param d_post_opt_offsets Per-slot optimized column offsets (can be nullptr for non-APC)
    /// @param cols_size sizeof(ColsType<uint8_t>) for the chip
    /// @param subs Column substitution table for APC
    /// @param apc_width APC width (0 for non-APC)
    /// @param calls_per_apc_row Number of chip calls packed per APC row (1 for non-APC)
    __device__ static RowSlice create_apc_aware(
        Fp *d_trace,
        size_t height,
        uint32_t idx,
        uint32_t *d_post_opt_offsets,
        size_t cols_size,
        uint32_t *subs,
        size_t apc_width,
        uint32_t calls_per_apc_row
    ) {
        bool is_apc = apc_width != 0;
        if (is_apc) {
            uint32_t slot = idx % calls_per_apc_row;
            size_t opt_offset = d_post_opt_offsets[slot];
            return RowSlice(
                d_trace + idx / calls_per_apc_row + opt_offset * height,
                height,
                opt_offset,
                cols_size * slot,
                subs,
                true
            );
        } else {
            return RowSlice(d_trace + idx, height);
        }
    }

    /// Overload that accepts ApcParams struct directly.
    __device__ static RowSlice create_apc_aware(
        Fp *d_trace,
        size_t height,
        uint32_t idx,
        size_t cols_size,
        const ApcParams &apc
    ) {
        return create_apc_aware(
            d_trace, height, idx, apc.post_opt_offsets,
            cols_size, apc.subs, apc.width, apc.calls_per_row
        );
    }

    __device__ __forceinline__ Fp &operator[](size_t column_index) const {
        // While implementing tracegen for SHA256, we encountered what we believe to be an nvcc
        // compiler bug. Occasionally, at various non-zero PTXAS optimization levels the compiler
        // tries to replace this multiplication with a series of SHL, ADD, and AND instructions
        // that we believe erroneously adds ~2^49 to the final address via an improper carry
        // propagation. To read more, see https://github.com/stephenh-axiom-xyz/cuda-illegal.
        return ptr[column_index * stride];
    }

    __device__ static RowSlice null() { return RowSlice(nullptr, 0, 0, 0, nullptr, false); }

    __device__ bool is_valid() const { return ptr != nullptr; }

    template <typename T>
    __device__ __forceinline__ void write(size_t column_index, T value) const {
        if (is_apc) {
            const uint32_t apc_idx = subs[dummy_offset + column_index];
            if (apc_idx != UINT32_MAX) {
                ptr[(apc_idx - optimized_offset) * stride] = value;
            }
        } else {
            ptr[column_index * stride] = value;
        }
    }

    template <typename T>
    __device__ __forceinline__ void write_array(size_t column_index, size_t length, const T *values)
        const {
        if (is_apc) {
#pragma unroll
            for (size_t i = 0; i < length; i++) {
                const uint32_t apc_idx = subs[dummy_offset + column_index + i];
                if (apc_idx != UINT32_MAX) {
                    ptr[(apc_idx - optimized_offset) * stride] = values[i];
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < length; i++) {
                ptr[(column_index + i) * stride] = values[i];
            }
        }
    }

    template <typename T>
    __device__ __forceinline__ void write_bits(size_t column_index, const T value) const {
#pragma unroll
        for (size_t i = 0; i < sizeof(T) * 8; i++) {
            ptr[(column_index + i) * stride] = (value >> i) & 1;
        }
    }

    __device__ __forceinline__ void fill_zero(size_t column_index_from, size_t length) const {
        if (is_apc) {
#pragma unroll
            for (size_t i = 0, c = column_index_from; i < length; i++, c++) {
                const uint32_t apc_idx = subs[dummy_offset + c];
                if (apc_idx != UINT32_MAX) {
                    ptr[(apc_idx - optimized_offset) * stride] = 0;
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0, c = column_index_from; i < length; i++, c++) {
                ptr[c * stride] = 0;
            }
        }
    }

    // Use the non-apc version regardless of whether the row is apc or not
    // Used for filling dummy rows in APC, so that we don't compute offsets but directly fill the buffer to zero
    __device__ __forceinline__ void fill_zero_no_offset(size_t column_index_from, size_t length) const {
#pragma unroll
            for (size_t i = 0, c = column_index_from; i < length; i++, c++) {
                ptr[c * stride] = 0;
            }
    }

    __device__ __forceinline__ RowSlice slice_from(size_t column_index) const {
        if (is_apc) {
            uint32_t gap = number_of_gaps_in(subs, dummy_offset, column_index);
            return RowSlice(ptr + (column_index - gap) * stride, stride, optimized_offset + column_index - gap, dummy_offset + column_index, subs, is_apc);
        } else {
            return RowSlice(ptr + column_index * stride, stride, 0, 0, nullptr, false);
        }
    }

    __device__ __forceinline__ RowSlice shift_row(size_t n) const {
        return RowSlice(ptr + n, stride, optimized_offset, dummy_offset, subs, is_apc);
    }
};

template <typename T>
__device__ __forceinline__ unsigned long long to_debug_uint(T value) {
    using Base = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<Base, Fp>) {
        return static_cast<unsigned long long>(value.asRaw());
    } else {
        return static_cast<unsigned long long>(value);
    }
}

/// Compute the 0-based column index of member `FIELD` within struct template `STRUCT<T>`,
/// by instantiating it as `STRUCT<uint8_t>` so that offsetof yields the element index.
#define COL_INDEX(STRUCT, FIELD) (offsetof(STRUCT<uint8_t>, FIELD))

/// Compute the fixed array length of `FIELD` within `STRUCT<T>`
#define COL_ARRAY_LEN(STRUCT, FIELD) (sizeof(static_cast<STRUCT<uint8_t> *>(nullptr)->FIELD))

/// Write a single value into `FIELD` of struct `STRUCT<T>` at a given row.
#define COL_WRITE_VALUE(ROW, STRUCT, FIELD, VALUE) (ROW).write(COL_INDEX(STRUCT, FIELD), VALUE)

/// Write an array of values into the fixed‐length `FIELD` array of `STRUCT<T>` for one row.
#define COL_WRITE_ARRAY(ROW, STRUCT, FIELD, VALUES)                                                \
    (ROW).write_array(COL_INDEX(STRUCT, FIELD), COL_ARRAY_LEN(STRUCT, FIELD), VALUES)

/// Write a single value bits into `FIELD` of struct `STRUCT<T>` at a given row.
#define COL_WRITE_BITS(ROW, STRUCT, FIELD, VALUE) (ROW).write_bits(COL_INDEX(STRUCT, FIELD), VALUE)

/// Fill entire `FIELD` of `STRUCT<T>` with zeros.
#define COL_FILL_ZERO(ROW, STRUCT, FIELD)                                                          \
    (ROW).fill_zero(                                                                               \
        COL_INDEX(STRUCT, FIELD), sizeof(static_cast<STRUCT<uint8_t> *>(nullptr)->FIELD)           \
    )

/// Fill dummy rows with zeros, handling both APC and non-APC cases.
/// For non-APC: fills cols_size bytes starting from column 0.
/// For APC: fills opt_widths[slot] bytes, but only if idx < height * calls_per_row
/// to avoid writing beyond the allocated buffer.
/// This version accepts ApcParams struct.
#define FILL_DUMMY_ROW_APC(row, cols_size, idx, height, apc)                                       \
    do {                                                                                           \
        if (!(apc).is_apc()) {                                                                     \
            (row).fill_zero(0, (cols_size));                                                       \
        } else if ((idx) < (height) * (apc).calls_per_row) {                                       \
            (row).fill_zero_no_offset(0, (apc).opt_widths[(idx) % (apc).calls_per_row]);           \
        }                                                                                          \
    } while(0)


__device__ __forceinline__ size_t number_of_gaps_in(const uint32_t *sub, size_t start, size_t len) {
    size_t gaps = 0;
#pragma unroll
    for (size_t i = start; i < start + len; ++i) {
        if (sub[i] == UINT32_MAX) {
            ++gaps;
        }
    }
    return gaps;
}
