#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_sampleselect.h"

namespace magma_sampleselect {

SYCL_EXTERNAL void reduce_counts(const int32_t *__restrict__ in,
                                 int32_t *__restrict__ out, int32_t num_blocks,
                                 sycl::nd_item<3> item_ct1) {
    int32_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (idx < searchtree_width) {
        int32_t sum{};
        for (int32_t i = 0; i < num_blocks; ++i) {
            sum += in[idx + i * searchtree_width];
        }
        out[idx] = sum;
    }
}

SYCL_EXTERNAL void prefix_sum_counts(int32_t *__restrict__ in,
                                     int32_t *__restrict__ out,
                                     int32_t num_blocks,
                                     sycl::nd_item<3> item_ct1) {
    int32_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (idx < searchtree_width) {
        int32_t sum{};
        for (int32_t i = 0; i < num_blocks; ++i) {
            auto tmp = in[idx + i * searchtree_width];
            in[idx + i * searchtree_width] = sum;
            sum += tmp;
        }
        out[idx] = sum;
    }
}

SYCL_EXTERNAL void sampleselect_findbucket(int32_t *__restrict__ totalcounts,
                                           int32_t rank,
                                           unsigned *__restrict__ out_bucket,
                                           int32_t *__restrict__ out_rank,
                                           sycl::nd_item<3> item_ct1,
                                           int32_t *sums) {
    prefix_sum_select(totalcounts, rank, out_bucket, out_rank, item_ct1, sums);
}

int32_t sampleselect_alloc_size(int32_t size) {
    return 1 // bucket index
         + 1 // rank
         + 1 // atomic
         + searchtree_width   // totalcount
         + num_grouped_blocks
           * searchtree_width // localcount
         + (size + 3) / 4;    // oracles
}

int32_t realloc_if_necessary(magma_ptr *ptr, magma_int_t *size, magma_int_t required_size) {
    int32_t info = 0;
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
