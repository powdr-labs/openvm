# CUDA Optimization Skill

Optimize CUDA kernel code for maximum GPU performance, based on the NVIDIA CUDA Programming Guide (Release 13.2) and CUDA C++ Best Practices Guide.

## When to Use

Activate this skill when:
- Reviewing or writing CUDA kernel code (.cu files)
- Optimizing GPU trace generation kernels
- Analyzing kernel launch configurations
- Debugging GPU performance issues
- Writing Rust FFI bindings to CUDA kernels (cuda_abi.rs, cuda.rs files)

## Repository CUDA Architecture

This codebase uses CUDA for GPU-accelerated STARK proof trace generation. Key patterns:

- **Kernel files**: `extensions/*/circuit/cuda/src/*.cu`, `crates/*/cuda/src/*.cu`
- **Rust FFI bindings**: `**/cuda_abi.rs` (extern "C" declarations)
- **GPU chip implementations**: `**/cuda.rs` (Rust `Chip<DenseRecordArena, GpuBackend>` impls)
- **Field type**: BabyBear (31-bit prime field), accessed as `F` via `openvm_cuda_backend::prelude::F`
- **Pattern**: Host copies execution records to device -> kernel generates trace matrix rows -> result stays on device
- **Memory management**: `DeviceBuffer<T>`, `DeviceMatrix<F>`, `MemCopyH2D`
- **Build system**: `openvm-cuda-builder` with feature flag `cuda`
- **Documentation**: `docs/repo/cuda.md`

## Optimization Checklist

When reviewing CUDA code, check each category below in priority order.

### 1. Memory Coalescing (Highest Impact)

**Rule**: Threads in a warp (32 threads) must access contiguous, aligned memory addresses.

- Global memory accesses are served in **32-byte sectors** within **128-byte cache lines**
- Ideal: thread `i` accesses `base + i * sizeof(element)` (stride-1 access)
- Stride-2 access wastes **50%** bandwidth; larger strides are worse
- Prefer **Structure-of-Arrays (SoA)** over Array-of-Structures (AoS)
- `cudaMalloc` returns **256-byte aligned** pointers
- Misaligned accesses crossing sector boundaries cause extra transactions (20%+ overhead)

**Check**: In trace generation kernels, verify that adjacent threads write to adjacent columns in the trace matrix. The `RowSlice` abstraction should handle this, but custom access patterns may not.

### 2. Shared Memory Bank Conflicts

**Rule**: Shared memory has **32 banks**, 4 bytes wide each. Successive 32-bit words map to successive banks.

- If multiple threads in a warp access **different addresses** in the **same bank**, accesses serialize
- **Broadcast**: All threads reading the **same** address = no conflict
- **Padding trick**: Use `__shared__ float tile[32][33]` instead of `[32][32]` to avoid stride conflicts
- 64-bit mode available via `cudaDeviceSetSharedMemConfig` (8-byte bank width)

**Check**: When shared memory is used as a scratchpad for field elements, ensure access patterns don't cause bank conflicts. BabyBear field elements are 4 bytes, so bank `= (byte_offset / 4) % 32`.

### 3. Occupancy and Launch Configuration

**Rule**: Block sizes must be multiples of **32** (warp size). Target **128, 256, or 512** threads per block.

**Hardware limits (key compute capabilities):**

| Parameter | CC 7.0 (V100) | CC 8.0 (A100) | CC 8.9 (Ada) | CC 9.0 (H100) |
|---|---|---|---|---|
| Max threads/SM | 2048 | 2048 | 1536 | 2048 |
| Max warps/SM | 64 | 64 | 48 | 64 |
| Max blocks/SM | 32 | 32 | 24 | 32 |
| Max threads/block | 1024 | 1024 | 1024 | 1024 |
| Registers/SM | 65536 | 65536 | 65536 | 65536 |
| Max registers/thread | 255 | 255 | 255 | 255 |
| Shared memory/SM | 96 KB | 164 KB | 100 KB | 228 KB |

**Occupancy limiters** (whichever is most restrictive):
1. Registers per thread (more registers -> fewer warps fit)
2. Shared memory per block (more shmem -> fewer blocks fit)
3. Threads per block (if block count * block size < max threads/SM)
4. Max blocks per SM hard cap

Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` to guide register allocation:
```cpp
__global__ void __launch_bounds__(256, 2) my_kernel() { ... }
```

Use `-Xptxas=-v` to check register and shared memory usage. Use `-Xptxas=-maxrregcount=N` to cap registers.

**Check**: Verify that `kernel_launch_params()` in this codebase produces sensible grid/block dimensions. Ensure no blocks are launched with 0 active threads.

### 4. Warp Divergence

**Rule**: When threads in a warp take different branch paths, both paths execute serially.

- All 32 threads execute the same instruction; divergent branches disable inactive threads
- Worst case: 50% throughput loss per if/else within a warp
- Restructure branches to align with warp boundaries when possible
- Use predication for short branches (compiler does this automatically for simple cases)
- **Independent thread scheduling** (CC 7.0+): threads have per-thread program counters, but divergence still reduces throughput

**Check**: In trace generation kernels, look for conditionals that vary per-thread within a warp. If threads handle different record types, consider sorting records or separating into different kernels.

### 5. Arithmetic Optimization

**Throughput per SM per clock cycle (Ampere/Hopper):**

| Operation | FP32 | FP64 | INT32 |
|---|---|---|---|
| Add/Mul/FMA | 128 | 64 | 64 |
| Special (sin/cos/exp) | 32 | -- | -- |

- **Fused Multiply-Add (FMA)**: `a * b + c` in a single instruction. Don't separate multiply and add.
- **Integer division/modulo** are expensive (~20 cycles). Use bitwise ops for powers of 2: `>> n` for `/2^n`, `& (n-1)` for `%2^n`. Compiler optimizes compile-time-known power-of-2 divisors.
- **Fast math intrinsics**: `__sinf()`, `__cosf()`, `__expf()`, `__fdividef()` are faster but less precise. `--use_fast_math` compiler flag enables globally.
- For BabyBear field arithmetic: modular reduction is the main cost. Look for opportunities to delay reductions (lazy reduction) and batch them.

### 6. Host-Device Transfer

**Rule**: Minimize transfers. PCIe bandwidth is orders of magnitude less than device memory.

| Interconnect | Bandwidth |
|---|---|
| PCIe 4.0 x16 | ~25 GB/s |
| PCIe 5.0 x16 | ~50 GB/s |
| Device memory (A100) | ~2 TB/s |
| Device memory (H100) | ~3.35 TB/s |

- Use **pinned (page-locked) memory**: `cudaMallocHost()` / `cudaHostAlloc()` for higher transfer rates and async capability
- **Overlap** transfers with computation using CUDA streams
- **Batch** small transfers into larger ones
- Keep intermediate data on device between kernel calls
- Don't transfer data just to inspect it on host during development (use device-side printf or assertions instead)

**Check**: In this codebase, `MemCopyH2D` copies records to device. Verify that trace matrices stay on device (`DeviceMatrix`) and aren't roundtripped unnecessarily.

### 7. Memory Hierarchy Usage

**Memory spaces reference:**

| Memory | Scope | Location | Latency | Size |
|---|---|---|---|---|
| Register | Thread | SM | ~0 (pipeline) | 255 per thread |
| Shared | Block | SM (on-chip) | ~20-30 cycles | 48-228 KB/SM |
| L1 Cache | SM | SM (on-chip) | ~30 cycles | Shared with shmem |
| L2 Cache | Device | On-chip | ~200 cycles | 6-72 MB |
| Global | Grid | DRAM | ~400-800 cycles | GB-scale |
| Constant | Grid | Cached | ~1 cycle (broadcast) | 64 KB total |
| Local | Thread | DRAM (cached) | ~400-800 cycles | 512 KB/thread |

- Use `const __restrict__` pointers to enable read-only cache path (`__ldg()` route)
- **Constant memory** (64 KB): excellent for lookup tables accessed uniformly by all threads (broadcast). Serializes if threads access different addresses.
- **Register spills to local memory** (DRAM!): watch for kernels using >32 registers per thread. Use `--res-usage` flag and `--Xptxas=-warn-spills`
- L2 cache set-aside (CC 8.0+): use `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` for frequently accessed data

### 8. Streams and Concurrency

- Use **non-default streams** for concurrent kernel execution and overlapped transfers
- Create streams with `cudaStreamNonBlocking` flag for true concurrency
- Use `cudaEventCreateWithFlags(cudaEventDisableTiming)` for synchronization-only events (less overhead)
- **CUDA Graphs** (CC 7.0+): capture and replay kernel launch sequences to reduce launch overhead. Useful for repeated identical launch patterns like multi-chip trace generation.
- Default stream (stream 0) serializes across all other streams unless per-thread default streams are enabled

### 9. Compiler and Build Optimizations

- Compile for **target architecture**: `-arch=sm_XX` (e.g., `sm_80` for A100, `sm_90` for H100)
- **Link-Time Optimization (LTO)**: `-dlto` flag recovers cross-file optimization for separate compilation
- `-extra-device-vectorization`: enables more aggressive vectorization
- `-DNDEBUG`: disable runtime assertions in release builds
- `-res-usage`: print register/shared memory/constant memory usage per kernel
- `-Xptxas=-warn-lmem-usage`: warn if local memory (register spills) is used
- `-Xptxas=-warn-spills`: warn if registers are spilled

### 10. Profiling Workflow

Use these tools to identify bottlenecks before optimizing:

1. **Nsight Systems** (`nsys profile`): System-level timeline. See kernel durations, transfer overlap, stream concurrency.
2. **Nsight Compute** (`ncu`): Kernel-level. Shows memory throughput %, compute throughput %, occupancy, warp stalls, cache hit rates, bank conflicts.

Key metrics to check:
- **SM Occupancy** (achieved vs theoretical) -- low = register/shmem pressure or small blocks
- **Memory Throughput** (% of peak DRAM bandwidth) -- high = memory-bound kernel
- **Compute Throughput** (% of peak FLOPS) -- high = compute-bound kernel
- **Warp Stall Reasons**: memory dependency, execution dependency, synchronization, etc.
- **L1/L2 Cache Hit Rates** -- low hit rates suggest poor locality
- **Shared Memory Bank Conflicts** -- visible in "shared store/load bank conflicts" metric

### 11. Warp-Level Primitives

Use warp shuffles to communicate between threads **without shared memory**:

```cpp
// Warp-level reduction (sum)
int val = thread_data;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
```

- `__shfl_sync(mask, val, srcLane)`: broadcast from specific lane
- `__shfl_up_sync(mask, val, delta)`: shift up within warp
- `__shfl_down_sync(mask, val, delta)`: shift down within warp
- `__shfl_xor_sync(mask, val, laneMask)`: XOR-based exchange (butterfly pattern)
- `__ballot_sync(mask, predicate)`: bitmask of which threads satisfy predicate
- `__popc(__ballot_sync(...))`: population count for warp-level voting

These are faster than shared memory for intra-warp communication and don't require `__syncthreads()`.

### 12. Common Anti-Patterns

**Avoid these:**
- Uncoalesced global memory access (scattered reads/writes)
- Shared memory bank conflicts from stride-32 access patterns
- Excessive register usage causing spills to local memory (DRAM)
- Thread divergence within hot loops
- Unnecessary `__syncthreads()` calls (each is a barrier)
- Launching kernels with fewer than 32 threads per block
- Atomic operations on highly contended global memory addresses
- Transferring data host<->device when it could stay on device
- Using `cudaDeviceSynchronize()` when stream-level sync suffices
- Using double precision on consumer GPUs (1/32 throughput vs FP32)

## Reference

Based on:
- [CUDA Programming Guide, Release 13.2](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) (March 2026)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
