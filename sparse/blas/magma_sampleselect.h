/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel
*/

#include <limits>
#include <type_traits>
#include "magmasparse_internal.h"
#undef max

namespace magma_sampleselect {

constexpr magma_int_t sample_size_log2 = 10;
constexpr magma_int_t sample_size = 1 << sample_size_log2;
constexpr magma_int_t searchtree_height = 8;
constexpr magma_int_t searchtree_width = 1 << searchtree_height;
constexpr magma_int_t searchtree_size = 2 * searchtree_width - 1;

constexpr magma_int_t warp_size_log2 = 5;
constexpr magma_int_t warp_size = 1 << warp_size_log2;
constexpr magma_int_t max_block_size_log2 = 9;
constexpr magma_int_t max_block_size = 1 << max_block_size_log2;

constexpr magma_int_t bitonic_cutoff_log2 = sample_size_log2;
constexpr magma_int_t bitonic_cutoff = 1 << bitonic_cutoff_log2;

constexpr auto block_size = max_block_size;
constexpr auto num_grouped_blocks = block_size;
constexpr auto num_threads = block_size * num_grouped_blocks;

constexpr unsigned full_mask = 0xffffffff;

/*
 * Forward declarations
 */
magma_int_t sampleselect_alloc_size(magma_int_t);
magma_int_t realloc_if_necessary(magma_ptr*,magma_int_t*,magma_int_t);

/*
 * Type-independent kernels
 */
__global__ void reduce_counts(const magma_int_t*,magma_int_t*,magma_int_t);
__global__ void prefix_sum_counts(magma_int_t*,magma_int_t*,magma_int_t);
__global__ void sampleselect_findbucket(magma_int_t*,magma_int_t,unsigned*,magma_int_t*);

/*
 * Type-dependent kernels
 */

__global__ void build_searchtree(const float*,float*,magma_int_t);
__global__ void build_searchtree(const double*,double*,magma_int_t);
__global__ void select_bitonic_basecase(float*,float*,magma_int_t,magma_int_t);
__global__ void select_bitonic_basecase(double*,double*,magma_int_t,magma_int_t);
__global__ void sampleselect(float*,float*,float*,magma_int_t*,magma_int_t,magma_int_t,float*);
__global__ void sampleselect(double*,double*,double*,magma_int_t*,magma_int_t,magma_int_t,double*);
__global__ void count_buckets(const float*,const float*,magma_int_t*,magma_int_t,magma_int_t);
__global__ void count_buckets(const double*,const double*,magma_int_t*,magma_int_t,magma_int_t);
__global__ void count_buckets_write(const float*,const float*,magma_int_t*,unsigned*,magma_int_t,magma_int_t);
__global__ void count_buckets_write(const double*,const double*,magma_int_t*,unsigned*,magma_int_t,magma_int_t);
__global__ void collect_bucket_indirect(const float*,const unsigned*,const magma_int_t*,float*,magma_int_t,unsigned*,magma_int_t*,magma_int_t);
__global__ void collect_bucket_indirect(const double*,const unsigned*,const magma_int_t*,double*,magma_int_t,unsigned*,magma_int_t*,magma_int_t);

__global__ void sampleselect_tailcall(float*,float*,float*,magma_int_t*,float*);
__global__ void sampleselect_tailcall(double*,double*,double*,magma_int_t*,double*);

/*
 * Type-independent helpers
 */

/*
 * Sampling
 */
__device__ inline magma_int_t uniform_pick_idx(magma_int_t idx, magma_int_t samplesize, magma_int_t size) {
    auto stride = size / samplesize;
    if (stride == 0) {
        return idx * size / samplesize;
    } else {
        return idx * stride + stride / 2;
    }
}

__device__ inline magma_int_t random_pick_idx(magma_int_t idx, magma_int_t samplesize, magma_int_t size) {
    // TODO
    return uniform_pick_idx(idx, samplesize, size);
}

__device__ inline magma_int_t prefix_popc(unsigned mask, magma_int_t shift) {
    return __popc(mask << (32 - shift));
    /* alternative:
    magma_int_t prefix_mask = (1 << shift) - 1;
    return __popc(mask & prefix_mask);
     */
}

__device__ inline magma_int_t warp_aggr_atomic_count_mask(magma_int_t* atomic, unsigned amask, unsigned mask) {
    auto lane_idx = threadIdx.x % warp_size;
    magma_int_t ofs{};
    if (lane_idx == 0) {
      ofs = atomicAdd((int *)atomic, __popc(mask));
    }
#if (__CUDACC_VER_MAJOR__ >= 9)
    ofs = __shfl_sync(amask, ofs, 0);
#else
    ofs = __shfl(ofs, 0);
#endif
    auto local_ofs = prefix_popc(mask, lane_idx);
    return ofs + local_ofs;
}

__device__ inline magma_int_t warp_aggr_atomic_count_predicate(magma_int_t* atomic, unsigned amask, bool predicate) {
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
__device__ inline void store_packed_bytes(unsigned* output, unsigned amask, unsigned byte, magma_int_t idx) {
    // pack 4 consecutive bytes into an integer
    unsigned result = byte;
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

__device__ inline unsigned load_packed_bytes(const unsigned* input, unsigned amask, magma_int_t idx) {
    auto char_idx = idx % 4;
    auto pack_idx = idx / 4;
    unsigned packed{};
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
__device__ inline void small_prefix_sum_upward(magma_int_t* data) {
    constexpr auto size = 1 << searchtree_height;
    auto idx = threadIdx.x;
    // upward phase: reduce
    // here we build an implicit reduction tree, overwriting values
    // the entry at the end of a power-of-two block stores the sum of this block
    // the block sizes are increased stepwise
    for (magma_int_t blocksize = 2; blocksize <= size; blocksize *= 2) {
        magma_int_t base_idx = idx * blocksize;
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

__device__ inline void small_prefix_sum_downward(magma_int_t* data) {
    constexpr auto size = 1 << searchtree_height;
    auto idx = threadIdx.x;
    // downward phase: build prefix sum
    // every right child stores the sum of its left sibling
    // every left child stores its own sum
    // thus we store zero at the root
    if (idx == 0) {
        data[size - 1] = 0;
    }
    for (magma_int_t blocksize = size; blocksize != 1; blocksize /= 2) {
        magma_int_t base_idx = idx * blocksize;
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

__device__ inline void small_prefix_sum(magma_int_t* data) {
    small_prefix_sum_upward(data);
    __syncthreads();
    small_prefix_sum_downward(data);
}

/*
 * Prefix sum selection
 */
__device__ inline void prefix_sum_select(const magma_int_t* counts, magma_int_t rank, unsigned* out_bucket, magma_int_t* out_rank) {
    constexpr auto size = 1 << searchtree_height;
    // first compute prefix sum of counts
    auto idx = threadIdx.x;
    __shared__ magma_int_t sums[size];
    sums[2 * idx] = counts[2 * idx];
    sums[2 * idx + 1] = counts[2 * idx + 1];
    small_prefix_sum(sums);
    __syncthreads();
    if (idx >= warp_size) {
        return;
    }
    // then determine which group of size step the element belongs to
    constexpr magma_int_t step = size / warp_size;
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
__device__ inline void blockwise_work(magma_int_t local_work, magma_int_t size, F function, Args... args) {
    auto stride = num_grouped_blocks * blockDim.x; // this should be a parameter for general cases
    auto base_idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (magma_int_t i = 0; i < local_work; ++i) {
        magma_int_t idx = base_idx + i * stride;
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
__device__ inline void blockwise_work_local(magma_int_t size, F function) {
    for (magma_int_t i = threadIdx.x; i < size; i += blockDim.x) {
        function(i);
    }
}

__device__ inline magma_int_t searchtree_entry(magma_int_t idx) {
    // determine the level by the node index
    // rationale: a complete binary tree with 2^k leaves has 2^k - 1 inner nodes
    magma_int_t lvl = 31 - __clz(idx + 1);      // == log2(idx + 1)
    magma_int_t step = searchtree_width >> lvl; // == n / 2^lvl
    magma_int_t lvl_idx = idx - (1 << lvl) + 1; // index within the level
    return lvl_idx * step + step / 2;
}

} // namespace magma_sampleselect
