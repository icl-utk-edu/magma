/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel

       @precisions normal z -> s d c
*/

#include <limits>
#include <type_traits>
#include <iostream>
#include "magmasparse_internal.h"
#undef max

#define PRECISION_z

namespace {

constexpr magma_int_t sample_size_log2 = 10;
constexpr magma_int_t sample_size = 1 << sample_size_log2;
constexpr magma_int_t searchtree_height = 8;
constexpr magma_int_t searchtree_width = 1 << searchtree_height;
constexpr magma_int_t searchtree_size = 2 * searchtree_width - 1;

constexpr magma_int_t warp_size_log2 = 5;
constexpr magma_int_t warp_size = 1 << warp_size_log2;
constexpr magma_int_t max_block_size_log2 = 9;
constexpr magma_int_t max_block_size = 1 << max_block_size_log2;

magma_int_t sampleselect_alloc_size(magma_int_t size);

template<typename T, bool shared>
__global__ void sampleselect(T* __restrict__ in,
                             T* __restrict__ tmp,
                             T* __restrict__ tree,
                             magma_int_t* __restrict__ count_tmp,
                             magma_int_t size, magma_int_t rank,
                             T* __restrict__ out);

constexpr auto block_size = max_block_size;
constexpr auto num_grouped_blocks = block_size;
constexpr auto num_threads = block_size * num_grouped_blocks;

constexpr unsigned full_mask = 0xffffffff;

/*
 * General helpers
 */
template<typename T>
__device__ void swap(T& a, T& b) {
    auto tmp = b;
    b = a;
    a = tmp;
}

template<typename T>
struct max_helper {
    // workaround for ::max being a __host__ function
    constexpr static T value = std::numeric_limits<T>::max();
};

template<typename T>
__global__ void set_zero(T* in, magma_int_t size) {
    if (threadIdx.x < size) {
        in[threadIdx.x] = 0;
    }
}

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

/*
 * Warp-aggregated atomics
 */
__device__ inline bool is_group_leader(unsigned mask) {
    return (__ffs(mask) - 1) == (threadIdx.x % warp_size);
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
        ofs = atomicAdd(atomic, __popc(mask));
    }
    ofs = __shfl_sync(amask, ofs, 0);
    auto local_ofs = prefix_popc(mask, lane_idx);
    return ofs + local_ofs;
}

__device__ inline magma_int_t warp_aggr_atomic_count_predicate(magma_int_t* atomic, unsigned amask, bool predicate) {
    auto mask = __ballot_sync(amask, predicate);
    return warp_aggr_atomic_count_mask(atomic, amask, mask);
}

/*
 * Unaligned byte storage
 */
__device__ inline void store_packed_bytes(unsigned* output, unsigned amask, unsigned byte, magma_int_t idx) {
    // pack 4 consecutive bytes magma_int_to an magma_int_teger
    unsigned result = byte;
    // ------00 -> ----1100
    result |= __shfl_xor_sync(amask, result, 1, 4) << 8;
    // ----1100 -> 33221100
    result |= __shfl_xor_sync(amask, result, 2, 4) << 16;
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
    packed = __shfl_sync(amask, packed, (pack_idx * 4) % warp_size, 4);
    packed >>= char_idx * 8;
    packed &= 255;
    return packed;
}

/*
 * Sorting
 */
template<typename T>
__device__ void sort2(T* in, magma_int_t i, magma_int_t j, bool odd) {
    auto ei = in[i];
    auto ej = in[j];
    if (odd != (ej < ei)) {
        in[i] = ej;
        in[j] = ei;
    }
}

template<magma_int_t log2_size, typename T>
__device__ void bitonic_sort(T* in) {
    magma_int_t idx = threadIdx.x;
    // idx has the form | high | low | where /low/ has /round/ bits
    for (magma_int_t round = 0; round < log2_size; ++round) {
        // the lowest bit of /high/ decides the sort order
        bool odd = idx & (1 << round);
        for (magma_int_t bit = 1 << round; bit != 0; bit >>= 1) {
            // idx has the form | upper | lower | where /lower/ initially
            // has /round/ bits and gradually shrink
            magma_int_t lower = idx & (bit - 1);
            magma_int_t upper = idx ^ lower;
            // we then sort the elements | upper | 0/1 | lower |
            magma_int_t sort_idx = lower | (upper << 1);
            if (bit >= warp_size) {
                __syncthreads();
            } else {
                __syncwarp();
            }
            if (idx < 1 << (log2_size - 1)) {
                sort2(in, sort_idx, sort_idx | bit, odd);
            }
        }
    }
}

constexpr magma_int_t bitonic_cutoff_log2 = 10;
constexpr magma_int_t bitonic_cutoff = 1 << bitonic_cutoff_log2;

template<typename T>
__global__ void select_bitonic_basecase(T* __restrict__ in, T* __restrict__ out, magma_int_t size, magma_int_t rank) {
    __shared__ T data[bitonic_cutoff];
    magma_int_t idx = threadIdx.x;
    data[threadIdx.x] = idx < size ? in[idx] : max_helper<T>::value;
    __syncthreads();
    bitonic_sort<bitonic_cutoff_log2>(data);
    __syncthreads();
    if (idx == 0) {
        *out = data[rank];
    }
}

/*
 * Prefix sum
 */
template<magma_int_t size_log2>
__device__ void small_prefix_sum_upward(magma_int_t* data) {
    constexpr auto size = 1 << size_log2;
    auto idx = threadIdx.x;
    // upward phase: reduce
    // here we build an implicit reduction tree, overwriting values
    // the entry at the end of a power-of-two block stores the sum of this block
    // the block sizes are increased stepwise
    for (magma_int_t blocksize = 2; blocksize <= size; blocksize *= 2) {
        magma_int_t base_idx = idx * blocksize;
        if (blocksize > warp_size || true) { //TODO rethink
            __syncthreads();
        } else {
            __syncwarp();
        }
        if (base_idx < size) {
            data[base_idx + blocksize - 1] += data[base_idx + blocksize / 2 - 1];
        }
    }
}

template<magma_int_t size_log2>
__device__ void small_prefix_sum_downward(magma_int_t* data) {
    constexpr auto size = 1 << size_log2;
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
        if (blocksize > warp_size || true) { //TODO rethink
            static_assert(size / warp_size <= warp_size, "insufficient synchronization");
            __syncthreads();
        } else {
            __syncwarp();
        }
        if (base_idx < size) {
            // we preserve the invariant for the next level
            auto r = data[base_idx + blocksize - 1];
            auto l = data[base_idx + blocksize / 2 - 1];
            data[base_idx + blocksize / 2 - 1] = r;
            data[base_idx + blocksize - 1] = l + r;
        }
    }
}

template<magma_int_t size_log2>
__device__ void small_prefix_sum(magma_int_t* data) {
    small_prefix_sum_upward<size_log2>(data);
    __syncthreads();
    small_prefix_sum_downward<size_log2>(data);
}

/*
 * Prefix sum selection
 */
template<magma_int_t size_log2>
__device__ void prefix_sum_select(const magma_int_t* counts, magma_int_t rank, unsigned* out_bucket, magma_int_t* out_rank) {
    constexpr auto size = 1 << size_log2;
    // first compute prefix sum of counts
    auto idx = threadIdx.x;
    __shared__ magma_int_t sums[size];
    sums[2 * idx] = counts[2 * idx];
    sums[2 * idx + 1] = counts[2 * idx + 1];
    small_prefix_sum<size_log2>(sums);
    __syncthreads();
    if (idx >= warp_size) {
        return;
    }
    // then determine which group of size step the element belongs to
    constexpr magma_int_t step = size / warp_size;
    static_assert(step <= warp_size, "need a third selection level");
    auto mask = __ballot_sync(full_mask, sums[(warp_size - idx - 1) * step] > rank);
    if (idx >= step) {
        return;
    }
    auto group = __clz(mask) - 1;
    // finally determine which bucket within the group the element belongs to
    auto base_idx = step * group;
    constexpr auto cur_mask = ((1u << (step - 1)) << 1) - 1;
    mask = __ballot_sync(cur_mask, sums[base_idx + (step - idx - 1)] > rank);
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
__device__ void blockwise_work(magma_int_t local_work, magma_int_t size, F function, Args... args) {
    auto stride = num_grouped_blocks * blockDim.x; // this should be a parameter for general cases
    auto base_idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (magma_int_t i = 0; i < local_work; ++i) {
        magma_int_t idx = base_idx + i * stride;
        auto amask = __ballot_sync(full_mask, idx < size);
        if (idx < size) {
            function(args..., idx, amask);
        }
    }
}

template<typename F>
__device__ void blockwise_work_local(magma_int_t size, F function) {
    for (magma_int_t i = threadIdx.x; i < size; i += blockDim.x) {
        function(i);
    }
}


__device__ __forceinline__ magma_int_t searchtree_entry(magma_int_t idx) {
    // determine the level by the node index
    // rationale: a complete binary tree with 2^k leaves has 2^k - 1 inner nodes
    magma_int_t lvl = 31 - __clz(idx + 1);      // == log2(idx + 1)
    magma_int_t step = searchtree_width >> lvl; // == n / 2^lvl
    magma_int_t lvl_idx = idx - (1 << lvl) + 1; // index within the level
    return lvl_idx * step + step / 2;
}

template<typename T>
__device__ __forceinline__ magma_int_t searchtree_traversal(const T* searchtree, T el, unsigned amask, unsigned& equal_mask) {
    magma_int_t i = 0;
    equal_mask = amask;
    auto root_splitter = searchtree[0];
    bool next_smaller = el < root_splitter;
    for (magma_int_t lvl = 0; lvl < searchtree_height; ++lvl) {
        bool smaller = next_smaller;
        i = 2 * i + 2 - smaller;
        next_smaller = el < searchtree[i];
        auto local_mask = __ballot_sync(amask, smaller) ^ (smaller - 1);
        equal_mask &= local_mask;
    }
    return i - (searchtree_width - 1);
}

template<typename T>
__global__ void build_searchtree(const T* __restrict__ in, T* __restrict__ out, magma_int_t size) {
    __shared__ T sample_buffer[sample_size];
    __shared__ T leaves[searchtree_width];
    auto idx = threadIdx.x;

    sample_buffer[idx] = in[random_pick_idx(idx, sample_size, size)];
    __syncthreads();
    bitonic_sort<sample_size_log2>(sample_buffer);
    __syncthreads();
    if (idx < searchtree_width) {
        leaves[idx] = sample_buffer[uniform_pick_idx(idx, searchtree_width, sample_size)];
        out[idx + searchtree_width - 1] = leaves[idx];
    }
    __syncthreads();
    if (idx < searchtree_width - 1) {
        out[idx] = leaves[searchtree_entry(idx)];
    }
}

template<typename BucketCallback, typename T>
__device__ __forceinline__ void ssss_impl(const T* __restrict__ in,
                                              const T* __restrict__ tree,
                                              magma_int_t size, magma_int_t workcount,
                                              BucketCallback bucket_cb) {
    __shared__ T local_tree[searchtree_size];

    // load searchtree magma_int_to shared memory
    blockwise_work_local(searchtree_size, [&](magma_int_t i) {
        local_tree[i] = tree[i];
    });
    __syncthreads();

    blockwise_work(workcount, size, [&](magma_int_t idx, unsigned amask) {
        unsigned equal_mask{};
        auto bucket_idx = searchtree_traversal(local_tree, in[idx], amask, equal_mask);
        // sum of block-wide counts
        bucket_cb(idx, bucket_idx, amask, equal_mask);
    });
}

template<typename T, bool shared, bool collfree, bool write>
__global__ void count_buckets(const T* __restrict__ in,
                              const T* __restrict__ tree,
                              magma_int_t* __restrict__ counts,
                              unsigned* __restrict__ oracles,
                              magma_int_t size, magma_int_t workcount) {
    __shared__ magma_int_t local_counts[searchtree_width];
    if (shared) {
        blockwise_work_local(searchtree_width, [&](magma_int_t i) {
            local_counts[i] = 0;
        });
        __syncthreads();
    }
    ssss_impl(in, tree, size, workcount, [&](magma_int_t idx, magma_int_t bucket, unsigned amask, unsigned mask) {
        if (write) {
            static_assert(searchtree_height <= 8, "can't pack bucket idx magma_int_to byte");
            store_packed_bytes(oracles, amask, bucket, idx);
        }
        magma_int_t add = collfree ? __popc(mask) : 1;
        if (!collfree || is_group_leader(mask)) {
            if (shared) {
                atomicAdd(&local_counts[bucket], add);
            } else {
                atomicAdd(&counts[bucket], add);
            }
        }
    });
    if (shared) {
        __syncthreads();
        // store the local counts grouped by block idx
        blockwise_work_local(searchtree_width, [&](magma_int_t i) {
            counts[i + blockIdx.x * searchtree_width] = local_counts[i];
        });
    }
}

__global__ void reduce_counts(const int* __restrict__ in,
                              int* __restrict__ out,
                              int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < searchtree_width) {
        int sum{};
        for (int i = 0; i < num_blocks; ++i) {
            sum += in[idx + i * searchtree_width];
        }
        out[idx] = sum;
    }
}

__global__ void prefix_sum_counts(magma_int_t* __restrict__ in,
                                  magma_int_t* __restrict__ out,
                                  magma_int_t num_blocks) {
    magma_int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < searchtree_width) {
        magma_int_t sum{};
        for (magma_int_t i = 0; i < num_blocks; ++i) {
            auto tmp = in[idx + i * searchtree_width];
            in[idx + i * searchtree_width] = sum;
            sum += tmp;
        }
        out[idx] = sum;
    }
}

template<typename T, bool shared>
__global__ void collect_bucket(const T* __restrict__ data,
                               const unsigned* __restrict__ oracles_packed,
                               const magma_int_t* __restrict__ prefix_sum,
                               T* __restrict__ out,
                               magma_int_t size, unsigned bucket,
                               magma_int_t* __restrict__ atomic, magma_int_t workcount) {
    __shared__ magma_int_t count;
    if (shared && threadIdx.x == 0) {
        count = prefix_sum[bucket + searchtree_width * blockIdx.x];
    }
    __syncthreads();
    blockwise_work(workcount, size, [&](magma_int_t idx, unsigned amask) {
            auto packed = load_packed_bytes(oracles_packed, amask, idx);
            magma_int_t ofs{};
            if (shared) {
                ofs = warp_aggr_atomic_count_predicate(&count, amask, packed == bucket);
            } else {
                ofs = warp_aggr_atomic_count_predicate(atomic, amask, packed == bucket);
            }
            if (packed == bucket) {
                out[ofs] = data[idx];
            }
        });
}

template<typename T, bool shared>
__global__ void sampleselect_tailcall(T* __restrict__ in, T* __restrict__ tmp, T* __restrict__ tree,
                                      magma_int_t* __restrict__ count_tmp, T* __restrict__ out);

__global__ void sampleselect_findbucket(magma_int_t* __restrict__ totalcounts, magma_int_t rank, unsigned* __restrict__ out_bucket, magma_int_t* __restrict__ out_rank) {
    prefix_sum_select<searchtree_height>(totalcounts, rank, out_bucket, out_rank);
}

template<typename T, bool shared>
__global__ void collect_bucket_indirect(const T* __restrict__ data,
                                        const unsigned* __restrict__ oracles_packed,
                                        const magma_int_t* __restrict__ prefix_sum,
                                        T* __restrict__ out,
                                        magma_int_t size, unsigned* bucket_ptr,
                                        magma_int_t* __restrict__ atomic, magma_int_t workcount) {
    __shared__ magma_int_t count;
    auto bucket = *bucket_ptr;
    if (shared && threadIdx.x == 0) {
        count = prefix_sum[bucket + searchtree_width * blockIdx.x];
    }
    __syncthreads();
    blockwise_work(workcount, size, [&](magma_int_t idx, unsigned amask) {
            auto packed = load_packed_bytes(oracles_packed, amask, idx);
            magma_int_t ofs{};
            if (shared) {
                ofs = warp_aggr_atomic_count_predicate(&count, amask, packed == bucket);
            } else {
                ofs = warp_aggr_atomic_count_predicate(atomic, amask, packed == bucket);
            }
            if (packed == bucket) {
                out[ofs] = data[idx];
            }
        });
}

magma_int_t sampleselect_alloc_size(magma_int_t size) {
    static_assert(sizeof(magma_int_t) == sizeof(unsigned), "c++ is broken");
    return 1 // bucket index
         + 1 // rank
         + 1 // atomic
         + searchtree_width   // totalcount
         + num_grouped_blocks
           * searchtree_width // localcount
         + (size + 3) / 4;    // oracles
}

template<typename T, bool shared>
__device__ void launch_sampleselect(T* __restrict__ in, T* __restrict__ tmp, T* __restrict__ tree,
                                    T* __restrict__ out, magma_int_t* __restrict__ count_tmp, magma_int_t size, magma_int_t rank) {
    if (threadIdx.x != 0) {
        return;
    }

    if (size <= bitonic_cutoff) {
        select_bitonic_basecase<<<1, bitonic_cutoff>>>(in, out, size, rank);
        return;
    }

    // launch kernels:
    // sample and build searchtree
    build_searchtree<T><<<1, sample_size>>>(in, tree, size);

    auto local_work = (size + num_threads - 1) / num_threads;
    auto bucket_idx = (unsigned*)count_tmp;
    auto rank_out = ((magma_int_t*)bucket_idx) + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;
    auto localcounts = totalcounts + searchtree_width;
    auto oracles = (unsigned*)(localcounts + num_grouped_blocks * searchtree_width);

    // count buckets
    if (shared) {
        count_buckets<T,1,1,1><<<num_grouped_blocks, block_size>>>(in, tree, localcounts, oracles, size, local_work);
        prefix_sum_counts<<<searchtree_width, num_grouped_blocks>>>(localcounts, totalcounts, num_grouped_blocks);
        sampleselect_findbucket<<<1, searchtree_width / 2>>>(totalcounts, rank, bucket_idx, rank_out);
        collect_bucket_indirect<T,1><<<num_grouped_blocks, block_size>>>(in, oracles, localcounts, tmp, size, bucket_idx, nullptr, local_work);
        sampleselect_tailcall<T,1><<<1, 1>>>(tmp, in, tree, count_tmp, out);
    } else {
        *atomic = 0;
        set_zero<<<1, searchtree_width>>>(totalcounts, searchtree_width);
        count_buckets<T,0,1,1><<<num_grouped_blocks, block_size>>>(in, tree, totalcounts, oracles, size, local_work);
        sampleselect_findbucket<<<1, searchtree_width / 2>>>(totalcounts, rank, bucket_idx, rank_out);
        collect_bucket_indirect<T,0><<<num_grouped_blocks, block_size>>>(in, oracles, nullptr, tmp, size, bucket_idx, atomic, local_work);
        sampleselect_tailcall<T,0><<<1, 1>>>(tmp, in, tree, count_tmp, out);
    }
}

template<typename T, bool shared>
__global__ void sampleselect_tailcall(T* __restrict__ in, T* __restrict__ tmp, T* __restrict__ tree,
                                      magma_int_t* __restrict__ count_tmp, T* __restrict__ out) {
    if (threadIdx.x != 0) {
        return;
    }
    auto bucket_idx = count_tmp;
    auto rank_out = bucket_idx + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;

    auto size = totalcounts[*bucket_idx];
    auto rank = *rank_out;
    launch_sampleselect<T,shared>(in, tmp, tree, out, count_tmp, size, rank);
}

template<typename T, bool shared>
__global__ void sampleselect(T* __restrict__ in, T* __restrict__ tmp, T* __restrict__ tree,
                             magma_int_t* __restrict__ count_tmp, magma_int_t size, magma_int_t rank, T* __restrict__ out) {
    launch_sampleselect<T,shared>(in, tmp, tree, out, count_tmp, size, rank);
}

template<typename T, typename U>
__global__ typename std::enable_if<std::is_arithmetic<T>::value>::type  compute_abs(const T* __restrict__ in, U* __restrict__ out, magma_int_t size) {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }

    auto v = in[idx];
    out[idx] = v * v;
}

template<typename T, typename U>
__global__ typename std::enable_if<!std::is_arithmetic<T>::value>::type compute_abs(const T* __restrict__ in, U* __restrict__ out, magma_int_t size) {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }

    auto v = in[idx];
    out[idx] = v.x * v.x + v.y * v.y;
}

magma_int_t realloc_if_necessary(magma_ptr *ptr, magma_int_t *size, magma_int_t required_size) {
    magma_int_t info = 0;
    std::cout << "realloc " << *ptr << " " << *size << " " << required_size;
    if (*size < required_size) {
        auto newsize = required_size * 5 / 4;
        std::cout << ", reallocating to " << newsize;
        CHECK(magma_free(*ptr));
        CHECK(magma_malloc(ptr, newsize));
        std::cout << ", resulting ptr " << *ptr;
        *size = newsize;
    }

cleanup:
    std::cout << std::endl;
    return info;
}

} // anonymous namespace

/**
    Purpose
    -------

    This routine selects a threshold separating the subset_size smallest
    magnitude elements from the rest.

    Arguments
    ---------

    @param[in]
    total_size  magma_int_t
                size of array val

    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate

    @param[in]
    val         magmaDoubleComplex
                array containing the values

    @param[out]
    thrs        double*
                computed threshold

    @param[inout]
    tmp_ptr     magma_ptr*
                pointer to pointer to temporary storage.
                May be reallocated during execution.

    @param[inout]
    tmp_size    magma_int_t*
                pointer to size of temporary storage.
                May be increased during execution.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsampleselect(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue )
{    
    magma_int_t info = 0;

    magma_int_t num_blocks = magma_ceildiv(total_size, block_size);
    magma_int_t required_size = sizeof(double) * (total_size * 2 + searchtree_size)
                                + sizeof(int) * sampleselect_alloc_size(total_size);
    auto realloc_result = realloc_if_necessary(tmp_ptr, tmp_size, required_size);

    double* gputmp1 = (double*)*tmp_ptr;
    double* gputmp2 = gputmp1 + total_size;
    double* gputree = gputmp2 + total_size;
    double* gpuresult = gputree + searchtree_size;
    magma_int_t* gpuints = (int*)(gpuresult + 1);

    CHECK(realloc_result);

    compute_abs<<<num_blocks, block_size, 0, queue->cuda_stream()>>>
        (val, gputmp1, total_size);
    sampleselect<double, 1><<<1, 1, 0, queue->cuda_stream()>>>
        (gputmp1, gputmp2, gputree, gpuints, total_size, subset_size, gpuresult);
    magma_dgetvector(1, gpuresult, 1, thrs, 1, queue );
    *thrs = std::sqrt(*thrs);

cleanup:
    return info;
}

/**
    Purpose
    -------

    This routine selects an approximate threshold separating the subset_size
    smallest magnitude elements from the rest.

    Arguments
    ---------

    @param[in]
    total_size  magma_int_t
                size of array val

    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate

    @param[in]
    val         magmaDoubleComplex
                array containing the values

    @param[out]
    thrs        double*
                computed threshold

    @param[inout]
    tmp_ptr     magma_ptr*
                pointer to pointer to temporary storage.
                May be reallocated during execution.

    @param[inout]
    tmp_size    magma_int_t*
                pointer to size of temporary storage.
                May be increased during execution.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsampleselect_approx(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    auto num_blocks = magma_ceildiv(total_size, block_size);
    auto local_work = (total_size + num_threads - 1) / num_threads;
    auto required_size = sizeof(double) * (total_size + searchtree_size)
                         + sizeof(int) * (searchtree_width * (num_grouped_blocks + 1) + 1);
    auto realloc_result = realloc_if_necessary(tmp_ptr, tmp_size, required_size);

    double* gputmp = (double*)*tmp_ptr;
    double* gputree = gputmp + total_size;
    unsigned* gpubucketidx = (unsigned*)(gputree + searchtree_size);
    magma_int_t* gpurankout = (magma_int_t*)(gpubucketidx + 1);
    magma_int_t* gpucounts = gpurankout + 1;
    magma_int_t* gpulocalcounts = gpucounts + searchtree_width;
    magma_int_t bucketidx{};

    CHECK(realloc_result);

    compute_abs<<<num_blocks, block_size, 0, queue->cuda_stream()>>>
        (val, gputmp, total_size);
    build_searchtree<double><<<1, sample_size, 0, queue->cuda_stream()>>>
        (gputmp, gputree, total_size);
    count_buckets<double,1,1,0><<<num_grouped_blocks, block_size, 0, queue->cuda_stream()>>>
        (gputmp, gputree, gpulocalcounts, nullptr, total_size, local_work);
    reduce_counts<<<searchtree_width, num_grouped_blocks, 0, queue->cuda_stream()>>>
        (gpulocalcounts, gpucounts, num_grouped_blocks);
    sampleselect_findbucket<<<1, searchtree_width / 2, 0, queue->cuda_stream()>>>
        (gpucounts, subset_size, gpubucketidx, gpurankout);
    magma_igetvector(1, (magma_int_t*)gpubucketidx, 1, &bucketidx, 1, queue);
    magma_dgetvector(1, gputree + searchtree_width - 1 + bucketidx, 1, thrs, 1, queue);
    *thrs = std::sqrt(*thrs);

cleanup:
    return info;
}
