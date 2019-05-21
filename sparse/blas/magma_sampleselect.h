/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel
*/

#include <limits>
#include "magmasparse_internal.h"
#undef max

namespace magma_sampleselect {

constexpr int32_t sample_size_log2 = 10;
constexpr int32_t sample_size = 1 << sample_size_log2;
constexpr int32_t searchtree_height = 8;
constexpr int32_t searchtree_width = 1 << searchtree_height;
constexpr int32_t searchtree_size = 2 * searchtree_width - 1;

constexpr int32_t warp_size_log2 = 5;
constexpr int32_t warp_size = 1 << warp_size_log2;
constexpr int32_t max_block_size_log2 = 9;
constexpr int32_t max_block_size = 1 << max_block_size_log2;

constexpr int32_t bitonic_cutoff_log2 = sample_size_log2;
constexpr int32_t bitonic_cutoff = 1 << bitonic_cutoff_log2;

constexpr auto block_size = max_block_size;
constexpr auto num_grouped_blocks = block_size;
constexpr auto num_threads = block_size * num_grouped_blocks;

constexpr uint32_t full_mask = 0xffffffff;

/*
 * Forward declarations
 */
int32_t sampleselect_alloc_size(int32_t);
int32_t realloc_if_necessary(magma_ptr*,magma_int_t*,magma_int_t);

/*
 * Type-independent kernels
 */
__global__ void reduce_counts(const int32_t*,int32_t*,int32_t);
__global__ void prefix_sum_counts(int32_t*,int32_t*,int32_t);
__global__ void sampleselect_findbucket(int32_t*,int32_t,uint32_t*,int32_t*);

/*
 * Type-dependent kernels
 */

__global__ void build_searchtree(const float*,float*,int32_t);
__global__ void build_searchtree(const double*,double*,int32_t);
__global__ void select_bitonic_basecase(float*,float*,int32_t,int32_t);
__global__ void select_bitonic_basecase(double*,double*,int32_t,int32_t);
__global__ void sampleselect(float*,float*,float*,int32_t*,int32_t,int32_t,float*);
__global__ void sampleselect(double*,double*,double*,int32_t*,int32_t,int32_t,double*);
__global__ void count_buckets(const float*,const float*,int32_t*,int32_t,int32_t);
__global__ void count_buckets(const double*,const double*,int32_t*,int32_t,int32_t);
__global__ void count_buckets_write(const float*,const float*,int32_t*,uint32_t*,int32_t,int32_t);
__global__ void count_buckets_write(const double*,const double*,int32_t*,uint32_t*,int32_t,int32_t);
__global__ void collect_bucket_indirect(const float*,const uint32_t*,const int32_t*,float*,int32_t,uint32_t*,int32_t*,int32_t);
__global__ void collect_bucket_indirect(const double*,const uint32_t*,const int32_t*,double*,int32_t,uint32_t*,int32_t*,int32_t);

__global__ void sampleselect_tailcall(float*,float*,float*,int32_t*,float*);
__global__ void sampleselect_tailcall(double*,double*,double*,int32_t*,double*);

/*
 * Type-independent helpers
 */

/*
 * Sampling
 */
__device__ inline int32_t uniform_pick_idx(int32_t idx, int32_t samplesize, int32_t size) {
    auto stride = size / samplesize;
    if (stride == 0) {
        return idx * size / samplesize;
    } else {
        return idx * stride + stride / 2;
    }
}

__device__ inline int32_t random_pick_idx(int32_t idx, int32_t samplesize, int32_t size) {
    // TODO
    return uniform_pick_idx(idx, samplesize, size);
}

__device__ inline int32_t prefix_popc(uint32_t mask, int32_t shift) {
    return __popc(mask << (32 - shift));
    /* alternative:
    int32_t prefix_mask = (1 << shift) - 1;
    return __popc(mask & prefix_mask);
     */
}

__device__ inline int32_t warp_aggr_atomic_count_mask(int32_t* atomic, uint32_t amask, uint32_t mask) {
    auto lane_idx = threadIdx.x % warp_size;
    int32_t ofs{};
    if (lane_idx == 0) {
      ofs = atomicAdd(atomic, __popc(mask));
    }
#if (__CUDACC_VER_MAJOR__ >= 9)
    ofs = __shfl_sync(amask, ofs, 0);
#else
    ofs = __shfl(ofs, 0);
#endif
    auto local_ofs = prefix_popc(mask, lane_idx);
    return ofs + local_ofs;
}

__device__ inline int32_t warp_aggr_atomic_count_predicate(int32_t* atomic, uint32_t amask, bool predicate) {
#if (__CUDACC_VER_MAJOR__ >= 9)
    auto mask = __ballot_sync(amask, predicate);
#else
    auto mask = __ballot(predicate) & amask;
#endif
    return warp_aggr_atomic_count_mask(atomic, amask, mask);
}

/*
 * Unaligned byte storage
 */
__device__ inline void store_packed_bytes(uint32_t* output, uint32_t amask, uint32_t byte, int32_t idx) {
    // pack 4 consecutive bytes into an integer
    uint32_t result = byte;
#if (__CUDACC_VER_MAJOR__ >= 9)
    // ------00 -> ----1100
    result |= __shfl_xor_sync(amask, result, 1, 4) << 8;
    // ----1100 -> 33221100
    result |= __shfl_xor_sync(amask, result, 2, 4) << 16;
#else
    // ------00 -> ----1100
    result |= __shfl_xor(result, 1, 4) << 8;
    // ----1100 -> 33221100
    result |= __shfl_xor(result, 2, 4) << 16;
#endif
    if (idx % 4 == 0) {
        output[idx / 4] = result;
    }
}

__device__ inline uint32_t load_packed_bytes(const uint32_t* input, uint32_t amask, int32_t idx) {
    auto char_idx = idx % 4;
    auto pack_idx = idx / 4;
    uint32_t packed{};
    // first thread in quartet loads the data
    if (char_idx == 0) {
        packed = input[pack_idx];
    }
    // distribute the data onto all threads
#if (__CUDACC_VER_MAJOR__ >= 9)
    packed = __shfl_sync(amask, packed, (pack_idx * 4) % warp_size, 4);
#else
    packed = __shfl(packed, (pack_idx * 4) % warp_size, 4);
#endif
    packed >>= char_idx * 8;
    packed &= 255;
    return packed;
}

/*
 * Prefix sum
 */
__device__ inline void small_prefix_sum_upward(int32_t* data) {
    constexpr auto size = 1 << searchtree_height;
    auto idx = threadIdx.x;
    // upward phase: reduce
    // here we build an implicit reduction tree, overwriting values
    // the entry at the end of a power-of-two block stores the sum of this block
    // the block sizes are increased stepwise
    for (int32_t blocksize = 2; blocksize <= size; blocksize *= 2) {
        int32_t base_idx = idx * blocksize;
#if (__CUDACC_VER_MAJOR__ >= 9)
        if (blocksize > warp_size || true) { //TODO rethink
            __syncthreads();
        } else {
            __syncwarp();
        }
#else
        __syncthreads();
#endif
        if (base_idx < size) {
            data[base_idx + blocksize - 1] += data[base_idx + blocksize / 2 - 1];
        }
    }
}

__device__ inline void small_prefix_sum_downward(int32_t* data) {
    constexpr auto size = 1 << searchtree_height;
    auto idx = threadIdx.x;
    // downward phase: build prefix sum
    // every right child stores the sum of its left sibling
    // every left child stores its own sum
    // thus we store zero at the root
    if (idx == 0) {
        data[size - 1] = 0;
    }
    for (int32_t blocksize = size; blocksize != 1; blocksize /= 2) {
        int32_t base_idx = idx * blocksize;
#if (__CUDACC_VER_MAJOR__ >= 9)
        if (blocksize > warp_size || true) { //TODO rethink
            static_assert(size / warp_size <= warp_size, "insufficient synchronization");
            __syncthreads();
        } else {
            __syncwarp();
        }
#else
        __syncthreads();
#endif
        if (base_idx < size) {
            // we preserve the invariant for the next level
            auto r = data[base_idx + blocksize - 1];
            auto l = data[base_idx + blocksize / 2 - 1];
            data[base_idx + blocksize / 2 - 1] = r;
            data[base_idx + blocksize - 1] = l + r;
        }
    }
}

__device__ inline void small_prefix_sum(int32_t* data) {
    small_prefix_sum_upward(data);
    __syncthreads();
    small_prefix_sum_downward(data);
}

/*
 * Prefix sum selection
 */
__device__ inline void prefix_sum_select(const int32_t* counts, int32_t rank, uint32_t* out_bucket, int32_t* out_rank) {
    constexpr auto size = 1 << searchtree_height;
    // first compute prefix sum of counts
    auto idx = threadIdx.x;
    __shared__ int32_t sums[size];
    sums[2 * idx] = counts[2 * idx];
    sums[2 * idx + 1] = counts[2 * idx + 1];
    small_prefix_sum(sums);
    __syncthreads();
    if (idx >= warp_size) {
        return;
    }
    // then determine which group of size step the element belongs to
    constexpr int32_t step = size / warp_size;
    static_assert(step <= warp_size, "need a third selection level");
#if (__CUDACC_VER_MAJOR__ >= 9)
    auto mask = __ballot_sync(full_mask, sums[(warp_size - idx - 1) * step] > rank);
#else
    auto mask = __ballot(sums[(warp_size - idx - 1) * step] > rank);
#endif
    if (idx >= step) {
        return;
    }
    auto group = __clz(mask) - 1;
    // finally determine which bucket within the group the element belongs to
    auto base_idx = step * group;
    constexpr auto cur_mask = ((1u << (step - 1)) << 1) - 1;
#if (__CUDACC_VER_MAJOR__ >= 9)
    mask = __ballot_sync(cur_mask, sums[base_idx + (step - idx - 1)] > rank);
#else
    mask = __ballot(sums[base_idx + (step - idx - 1)] > rank) & cur_mask;
#endif
    // here we need to subtract warp_size - step since we only use a subset of the warp
    if (idx == 0) {
        *out_bucket = __clz(mask) - 1 - (warp_size - step) + base_idx;
        *out_rank = rank - sums[*out_bucket];
    }
}

/*
 * Work assignment
 */
template<typename F, typename... Args>
__device__ inline void blockwise_work(int32_t local_work, int32_t size, F function, Args... args) {
    auto stride = num_grouped_blocks * blockDim.x; // this should be a parameter for general cases
    auto base_idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t i = 0; i < local_work; ++i) {
        int32_t idx = base_idx + i * stride;
#if (__CUDACC_VER_MAJOR__ >= 9)
        auto amask = __ballot_sync(full_mask, idx < size);
#else
        auto amask = __ballot(idx < size);
#endif
        if (idx < size) {
            function(args..., idx, amask);
        }
    }
}

template<typename F>
__device__ inline void blockwise_work_local(int32_t size, F function) {
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        function(i);
    }
}

__device__ inline int32_t searchtree_entry(int32_t idx) {
    // determine the level by the node index
    // rationale: a complete binary tree with 2^k leaves has 2^k - 1 inner nodes
    int32_t lvl = 31 - __clz(idx + 1);      // == log2(idx + 1)
    int32_t step = searchtree_width >> lvl; // == n / 2^lvl
    int32_t lvl_idx = idx - (1 << lvl) + 1; // index within the level
    return lvl_idx * step + step / 2;
}

} // namespace magma_sampleselect
