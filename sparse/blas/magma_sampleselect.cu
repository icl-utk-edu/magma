#include "magma_sampleselect.h"

namespace magma_sampleselect {

__global__ void reduce_counts(const magma_int_t* __restrict__ in,
                              magma_int_t* __restrict__ out,
                              magma_int_t num_blocks) {
    magma_int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < searchtree_width) {
        magma_int_t sum{};
        for (magma_int_t i = 0; i < num_blocks; ++i) {
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

__global__ void sampleselect_findbucket(magma_int_t* __restrict__ totalcounts,
                                        magma_int_t rank,
                                        unsigned* __restrict__ out_bucket,
                                        magma_int_t* __restrict__ out_rank) {
    prefix_sum_select(totalcounts, rank, out_bucket, out_rank);
}

magma_int_t sampleselect_alloc_size(magma_int_t size) {
    static_assert(sizeof(magma_int_t) >= sizeof(unsigned), "c++ is broken");
    return 1 // bucket index
         + 1 // rank
         + 1 // atomic
         + searchtree_width   // totalcount
         + num_grouped_blocks
           * searchtree_width // localcount
         + (size + 3) / 4;    // oracles
}

magma_int_t realloc_if_necessary(magma_ptr *ptr, magma_int_t *size, magma_int_t required_size) {
    magma_int_t info = 0;
    if (*size < required_size) {
        auto newsize = required_size * 5 / 4;
        CHECK(magma_free(*ptr));
        CHECK(magma_malloc(ptr, newsize));
        *size = newsize;
    }

cleanup:
    return info;
}

} // namespace magma_sampleselect