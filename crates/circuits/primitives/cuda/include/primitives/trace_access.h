#pragma once

#include "fp.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

/// APC parameters passed from Rust to CUDA.
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

};

/// Count eliminated columns (gaps) in a range of the APC substitution array.
/// In APC mode, columns marked with UINT32_MAX were eliminated during optimization.
/// This function counts how many such gaps exist in [start, start+len).
/// Used by slice_from() to compute the correct pointer offset in the optimized trace.
/// @param sub  Pointer to the APC substitution array
/// @param start  Starting index in the substitution array
/// @param len  Number of entries to scan
/// @return Number of entries equal to UINT32_MAX in the range
__device__ __forceinline__ size_t number_of_gaps_in(const uint32_t *sub, size_t start, size_t len);

/// A RowSlice is a contiguous section of a row in col-based trace.
/// Supports both direct trace access (non-APC) and APC-aware access with column substitution.
///
/// ## APC Mode
/// When `is_apc` is true, the trace has been optimized to remove redundant columns. The `subs`
/// array maps original column indices to their optimized positions (or UINT32_MAX if eliminated).
/// Writes to eliminated columns are ignored.
///
/// ## Non-APC Mode
/// When `is_apc` is false, columns map directly to trace positions without substitution.
struct RowSlice {
    /// Base pointer into the trace buffer, pointing to the first column of this row slice.
    Fp *ptr;

    /// Distance (in Fp elements) between consecutive columns in the trace.
    /// Equal to the trace height in column-major layout.
    size_t stride;

    /// Column offset in the optimized (APC) trace where this slice begins.
    /// Used to convert APC substitution indices back to ptr-relative offsets.
    /// Always 0 in non-APC mode.
    size_t optimized_offset;

    /// Column offset in the original (non-optimized) column layout.
    /// Used as base index into the `subs` substitution array.
    /// Always 0 in non-APC mode.
    size_t dummy_offset;

    /// Pointer to the APC substitution array mapping original column indices to optimized indices.
    /// Entry value UINT32_MAX indicates the column was eliminated and writes should be skipped.
    /// nullptr in non-APC mode.
    uint32_t *subs;

    /// Whether this RowSlice operates in APC mode with column substitution.
    bool is_apc;

    // Full constructor for APC-aware access
    __device__ RowSlice(Fp *ptr, size_t stride, size_t optimized_offset, size_t dummy_offset, uint32_t *subs, bool is_apc) : ptr(ptr), stride(stride), optimized_offset(optimized_offset), dummy_offset(dummy_offset), subs(subs), is_apc(is_apc) {}

    // Simple constructor for backward compatibility (non-APC mode)
    __device__ RowSlice(Fp *ptr, size_t stride) : ptr(ptr), stride(stride), optimized_offset(0), dummy_offset(0), subs(nullptr), is_apc(false) {}

    /// Create a RowSlice with APC-aware setup.
    /// For APC mode: if idx >= num_records, fills the dummy row and returns null.
    /// For non-APC mode: returns a valid RowSlice (caller handles dummy rows).
    /// @param d_trace Base pointer to the trace buffer
    /// @param height Trace height (stride between columns)
    /// @param idx Thread index (blockIdx.x * blockDim.x + threadIdx.x)
    /// @param cols_size sizeof(ColsType<uint8_t>) for the chip
    /// @param apc APC parameters struct
    /// @param num_records Number of real records to process
    __device__ static RowSlice create_apc_aware(
        Fp *d_trace,
        size_t height,
        uint32_t idx,
        size_t cols_size,
        const ApcParams &apc,
        size_t num_records
    ) {
        if (apc.is_apc()) {
            // Beyond APC buffer - this should never happen if kernel launch is correct
            assert(idx < height * apc.calls_per_row && "idx exceeds APC buffer bounds");

            uint32_t slot = idx % apc.calls_per_row;
            size_t opt_offset = apc.post_opt_offsets[slot];
            RowSlice row(
                d_trace + idx / apc.calls_per_row + opt_offset * height,
                height,
                opt_offset,
                cols_size * slot,
                apc.subs,
                true
            );

            // Dummy row - fill zeros and return null
            if (idx >= num_records) {
                row.fill_zero_no_offset(0, apc.opt_widths[slot]);
                return RowSlice::null();
            }

            return row;
        } else {
            return RowSlice(d_trace + idx, height);
        }
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
#pragma unroll
        for (size_t i = 0; i < length; i++) {
            write(column_index + i, values[i]);
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
#pragma unroll
        for (size_t i = 0, c = column_index_from; i < length; i++, c++) {
            write(c, 0);
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
        assert(!is_apc && "shift_row is only valid in non-APC mode");
        return RowSlice(ptr + n, stride, 0, 0, nullptr, false);
    }
};

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
