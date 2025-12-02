#pragma once

#include "fp.h"
#include "primitives/row_print_buffer.cuh"
#include <cstddef>
#include <cstdint>
#include <type_traits>


__device__ __forceinline__ size_t number_of_gaps_in(const uint32_t *sub, size_t start, size_t len);

/// A RowSlice is a contiguous section of a row in col-based trace.
struct RowSliceNew {
    Fp *ptr;
    size_t stride;
    size_t optimized_offset;
    size_t dummy_offset;
    uint32_t *subs;
    bool is_apc;


    __device__ RowSliceNew(Fp *ptr, size_t stride, size_t optimized_offset, size_t dummy_offset, uint32_t *subs, bool is_apc) : ptr(ptr), stride(stride), optimized_offset(optimized_offset), dummy_offset(dummy_offset), subs(subs), is_apc(is_apc) {}

    __device__ __forceinline__ Fp &operator[](size_t column_index) const {
        // While implementing tracegen for SHA256, we encountered what we believe to be an nvcc
        // compiler bug. Occasionally, at various non-zero PTXAS optimization levels the compiler
        // tries to replace this multiplication with a series of SHL, ADD, and AND instructions
        // that we believe erroneously adds ~2^49 to the final address via an improper carry
        // propagation. To read more, see https://github.com/stephenh-axiom-xyz/cuda-illegal.
        return ptr[column_index * stride];
    }

    __device__ static RowSliceNew null() { return RowSliceNew(nullptr, 0, 0, 0, nullptr, false); }

    __device__ bool is_valid() const { return ptr != nullptr; }

    template <typename T>
    __device__ __forceinline__ void write(size_t column_index, T value) const {
        ptr[column_index * stride] = value;
    }

    template <typename T>
    __device__ __forceinline__ void write_new(size_t column_index, T value) const {
        const uint32_t apc_idx = subs[dummy_offset + column_index];
        if (apc_idx != UINT32_MAX) {
            ptr[(apc_idx - optimized_offset) * stride] = value;
        }
    }

    template <typename T>
    __device__ __forceinline__ void write_array(size_t column_index, size_t length, const T *values)
        const {
#pragma unroll
        for (size_t i = 0; i < length; i++) {
            ptr[(column_index + i) * stride] = values[i];
        }
    }

    template <typename T>
    __device__ __forceinline__ void write_array_new(size_t column_index, size_t length, const T *values)
        const {
#pragma unroll
        for (size_t i = 0; i < length; i++) {
            const uint32_t apc_idx = subs[dummy_offset + column_index + i];
            if (apc_idx != UINT32_MAX) {
                ptr[(apc_idx - optimized_offset) * stride] = values[i];
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
#pragma unroll
        for (size_t i = 0, c = column_index_from; i < length; i++, c++) {
            ptr[c * stride] = 0;
        }
    }

    __device__ __forceinline__ RowSliceNew slice_from(size_t column_index) const {
        uint32_t gap = number_of_gaps_in(subs, dummy_offset, column_index);
        return RowSliceNew(ptr + (column_index - gap) * stride, stride, optimized_offset + column_index - gap, dummy_offset + column_index, subs, is_apc);
    }

    __device__ __forceinline__ RowSliceNew shift_row(size_t n) const {
        return RowSliceNew(ptr + n, stride, optimized_offset, dummy_offset, subs, is_apc);
    }
};

/// A RowSlice is a contiguous section of a row in col-based trace.
struct RowSlice {
    Fp *ptr;
    size_t stride;

    __device__ RowSlice(Fp *ptr, size_t stride) : ptr(ptr), stride(stride) {}

    __device__ __forceinline__ Fp &operator[](size_t column_index) const {
        // While implementing tracegen for SHA256, we encountered what we believe to be an nvcc
        // compiler bug. Occasionally, at various non-zero PTXAS optimization levels the compiler
        // tries to replace this multiplication with a series of SHL, ADD, and AND instructions
        // that we believe erroneously adds ~2^49 to the final address via an improper carry
        // propagation. To read more, see https://github.com/stephenh-axiom-xyz/cuda-illegal.
        return ptr[column_index * stride];
    }

    __device__ static RowSlice null() { return RowSlice(nullptr, 0); }

    __device__ bool is_valid() const { return ptr != nullptr; }

    template <typename T>
    __device__ __forceinline__ void write(size_t column_index, T value) const {
        ptr[column_index * stride] = value;
    }

    template <typename T>
    __device__ __forceinline__ void write_array(size_t column_index, size_t length, const T *values)
        const {
#pragma unroll
        for (size_t i = 0; i < length; i++) {
            ptr[(column_index + i) * stride] = values[i];
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
            ptr[c * stride] = 0;
        }
    }

    __device__ __forceinline__ RowSlice slice_from(size_t column_index) const {
        return RowSlice(ptr + column_index * stride, stride);
    }

    __device__ __forceinline__ RowSlice shift_row(size_t n) const {
        return RowSlice(ptr + n, stride);
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

/// Write a single value into `FIELD` of struct `STRUCT<T>` at a given row.                    
#define COL_WRITE_VALUE_NEW(ROW, STRUCT, FIELD, VALUE) (ROW).write_new(COL_INDEX(STRUCT, FIELD), VALUE)

/// Write an array of values into the fixed‐length `FIELD` array of `STRUCT<T>` for one row.
#define COL_WRITE_ARRAY(ROW, STRUCT, FIELD, VALUES)                                                \
    (ROW).write_array(COL_INDEX(STRUCT, FIELD), COL_ARRAY_LEN(STRUCT, FIELD), VALUES)

/// Write an array of values into the fixed‐length `FIELD` array of `STRUCT<T>` for one row.
#define COL_WRITE_ARRAY_NEW(ROW, STRUCT, FIELD, VALUES)                                                \
    (ROW).write_array_new(COL_INDEX(STRUCT, FIELD), COL_ARRAY_LEN(STRUCT, FIELD), VALUES)

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
