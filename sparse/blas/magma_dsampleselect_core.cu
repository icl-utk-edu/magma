/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel

       @precisions normal d -> s
*/

#include "magma_sampleselect.h"

#define PRECISION_d

namespace magma_sampleselect {

constexpr auto max_value = std::numeric_limits<double>::max();

/*
 * Sorting
 */
__device__ void sort2(double* in, magma_int_t i, magma_int_t j, bool odd) {
    auto ei = in[i];
    auto ej = in[j];
    if (odd != (ej < ei)) {
        in[i] = ej;
        in[j] = ei;
    }
}

__device__ void bitonic_sort(double* in) {
    magma_int_t idx = threadIdx.x;
    // idx has the form | high | low | where /low/ has /round/ bits
    for (magma_int_t round = 0; round < bitonic_cutoff_log2; ++round) {
        // the lowest bit of /high/ decides the sort order
        bool odd = idx & (1 << round);
        for (magma_int_t bit = 1 << round; bit != 0; bit >>= 1) {
            // idx has the form | upper | lower | where /lower/ initially
            // has /round/ bits and gradually shrink
            magma_int_t lower = idx & (bit - 1);
            magma_int_t upper = idx ^ lower;
            // we then sort the elements | upper | 0/1 | lower |
            magma_int_t sort_idx = lower | (upper << 1);
#if (__CUDACC_VER_MAJOR__ >= 9)
            if (bit >= warp_size) {
                __syncthreads();
            } else {
                __syncwarp();
            }
#else
            __syncthreads();
#endif
            if (idx < 1 << (bitonic_cutoff_log2 - 1)) {
                sort2(in, sort_idx, sort_idx | bit, odd);
            }
        }
    }
}

__global__ void select_bitonic_basecase(double* __restrict__ in, double* __restrict__ out, magma_int_t size, magma_int_t rank) {
    __shared__ double data[bitonic_cutoff];
    magma_int_t idx = threadIdx.x;
    data[threadIdx.x] = idx < size ? in[idx] : max_value;
    __syncthreads();
    bitonic_sort(data);
    __syncthreads();
    if (idx == 0) {
        *out = data[rank];
    }
}

__device__ magma_int_t searchtree_traversal(const double* searchtree, double el, unsigned amask, unsigned& equal_mask) {
    magma_int_t i = 0;
    equal_mask = amask;
    auto root_splitter = searchtree[0];
    bool next_smaller = el < root_splitter;
    for (magma_int_t lvl = 0; lvl < searchtree_height; ++lvl) {
        bool smaller = next_smaller;
        i = 2 * i + 2 - smaller;
        next_smaller = el < searchtree[i];
#if (__CUDACC_VER_MAJOR__ >= 9)
        auto local_mask = __ballot_sync(amask, smaller) ^ (smaller - 1);
#else
        auto local_mask = (__ballot(smaller) & amask) ^ (smaller - 1);
#endif
        equal_mask &= local_mask;
    }
    return i - (searchtree_width - 1);
}

__global__ void build_searchtree(const double* __restrict__ in, double* __restrict__ out, magma_int_t size) {
    __shared__ double sample_buffer[sample_size];
    __shared__ double leaves[searchtree_width];
    auto idx = threadIdx.x;

    sample_buffer[idx] = in[random_pick_idx(idx, sample_size, size)];
    __syncthreads();
    bitonic_sort(sample_buffer);
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

template<typename BucketCallback>
__device__ void ssss_impl(const double* __restrict__ in,
                          const double* __restrict__ tree,
                          magma_int_t size, magma_int_t workcount,
                          BucketCallback bucket_cb) {
    __shared__ double local_tree[searchtree_size];

    // load searchtree into shared memory
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

template<bool write>
__global__ void count_buckets(const double* __restrict__ in,
                              const double* __restrict__ tree,
                              magma_int_t* __restrict__ counts,
                              unsigned* __restrict__ oracles,
                              magma_int_t size, magma_int_t workcount) {
    __shared__ magma_int_t local_counts[searchtree_width];

    blockwise_work_local(searchtree_width, [&](magma_int_t i) {
        local_counts[i] = 0;
    });
    __syncthreads();
    ssss_impl(in, tree, size, workcount, [&](magma_int_t idx, magma_int_t bucket, unsigned amask, unsigned mask) {
        if (write) {
            static_assert(searchtree_height <= 8, "can't pack bucket idx into byte");
            store_packed_bytes(oracles, amask, bucket, idx);
        }
        atomicAdd(&local_counts[bucket], 1);
    });
    __syncthreads();
    // store the local counts grouped by block idx
    blockwise_work_local(searchtree_width, [&](magma_int_t i) {
        counts[i + blockIdx.x * searchtree_width] = local_counts[i];
    });
}

template __global__ void count_buckets<0>(const double*,const double*,magma_int_t*,unsigned*,magma_int_t,magma_int_t);
template __global__ void count_buckets<1>(const double*,const double*,magma_int_t*,unsigned*,magma_int_t,magma_int_t);

__global__ void collect_bucket_indirect(const double* __restrict__ data,
                                        const unsigned* __restrict__ oracles_packed,
                                        const magma_int_t* __restrict__ prefix_sum,
                                        double* __restrict__ out,
                                        magma_int_t size, unsigned* bucket_ptr,
                                        magma_int_t* __restrict__ atomic, magma_int_t workcount) {
    __shared__ magma_int_t count;
    auto bucket = *bucket_ptr;
    if (threadIdx.x == 0) {
        count = prefix_sum[bucket + searchtree_width * blockIdx.x];
    }
    __syncthreads();
    blockwise_work(workcount, size, [&](magma_int_t idx, unsigned amask) {
            auto packed = load_packed_bytes(oracles_packed, amask, idx);
            magma_int_t ofs{};
            ofs = warp_aggr_atomic_count_predicate(&count, amask, packed == bucket);
            if (packed == bucket) {
                out[ofs] = data[idx];
            }
        });
}

__device__ void launch_sampleselect(double* __restrict__ in, double* __restrict__ tmp, double* __restrict__ tree,
                                    double* __restrict__ out, magma_int_t* __restrict__ count_tmp, magma_int_t size, magma_int_t rank) {
    if (threadIdx.x != 0) {
        return;
    }

    if (size <= bitonic_cutoff) {
        select_bitonic_basecase<<<1, bitonic_cutoff>>>(in, out, size, rank);
        return;
    }

    // launch kernels:
    // sample and build searchtree
    build_searchtree<<<1, sample_size>>>(in, tree, size);

    auto local_work = (size + num_threads - 1) / num_threads;
    auto bucket_idx = (unsigned*)count_tmp;
    auto rank_out = ((magma_int_t*)bucket_idx) + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;
    auto localcounts = totalcounts + searchtree_width;
    auto oracles = (unsigned*)(localcounts + num_grouped_blocks * searchtree_width);

    // count buckets
    count_buckets<true><<<num_grouped_blocks, block_size>>>(in, tree, localcounts, oracles, size, local_work);
    prefix_sum_counts<<<searchtree_width, num_grouped_blocks>>>(localcounts, totalcounts, num_grouped_blocks);
    sampleselect_findbucket<<<1, searchtree_width / 2>>>(totalcounts, rank, bucket_idx, rank_out);
    collect_bucket_indirect<<<num_grouped_blocks, block_size>>>(in, oracles, localcounts, tmp, size, bucket_idx, nullptr, local_work);
    sampleselect_tailcall<<<1, 1>>>(tmp, in, tree, count_tmp, out);
}

__global__ void sampleselect_tailcall(double* __restrict__ in, double* __restrict__ tmp, double* __restrict__ tree,
                                      magma_int_t* __restrict__ count_tmp, double* __restrict__ out) {
    if (threadIdx.x != 0) {
        return;
    }
    auto bucket_idx = count_tmp;
    auto rank_out = bucket_idx + 1;
    auto atomic = rank_out + 1;
    auto totalcounts = atomic + 1;

    auto size = totalcounts[*bucket_idx];
    auto rank = *rank_out;
    launch_sampleselect(in, tmp, tree, out, count_tmp, size, rank);
}

__global__ void sampleselect(double* __restrict__ in, double* __restrict__ tmp, double* __restrict__ tree,
                             magma_int_t* __restrict__ count_tmp, magma_int_t size, magma_int_t rank, double* __restrict__ out) {
    launch_sampleselect(in, tmp, tree, out, count_tmp, size, rank);
}

} // namespace magma_sampleselect