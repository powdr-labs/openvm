#pragma once

#include <cstdio>

// Utility buffer to print a single APC row atomically from device code.
struct RowPrintBuffer {
    static constexpr int kCapacity = 8192;
    char data[kCapacity];
    int len;

    __device__ __forceinline__ void reset() { len = 0; }

    __device__ __forceinline__ void append_char(char c) {
        if (len < kCapacity - 1) {
            data[len++] = c;
        }
    }

    __device__ __forceinline__ void append_literal(const char *literal) {
        for (const char *ptr = literal; *ptr != '\0'; ++ptr) {
            append_char(*ptr);
        }
    }

    __device__ __forceinline__ void append_uint(unsigned long long value) {
        char tmp[32];
        int tmp_len = 0;

        if (value == 0) {
            tmp[tmp_len++] = '0';
        } else {
            while (value > 0 && tmp_len < static_cast<int>(sizeof(tmp))) {
                tmp[tmp_len++] = static_cast<char>('0' + (value % 10));
                value /= 10;
            }
        }

        for (int i = tmp_len - 1; i >= 0; --i) {
            append_char(tmp[i]);
        }
    }

    __device__ __forceinline__ void flush() {
        data[len] = '\0';
        printf("%s", data);
    }

    // Execute `fn` with this buffer after clearing it, then flush.
    // `fn` must be a device callable accepting `RowPrintBuffer &`.
    template <typename Fn>
    __device__ __forceinline__ void write_with(Fn fn) {
        reset();
        fn(*this);
        flush();
    }
};
