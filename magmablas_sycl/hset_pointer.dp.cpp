/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
       
       dedicated src for pointer arithmetic in fp16
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define PRECISION_h

void kernel_hset_pointer(
    magmaHalf **output_array,
    magmaHalf *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset, sycl::nd_item<3> item_ct1)
{
    output_array[item_ct1.get_group(2)] =
        input + item_ct1.get_group(2) * batch_offset + row + column * lda;
}

extern "C"
void magma_hset_pointer(
    magmaHalf **output_array,
    magmaHalf *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue)
{
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount),
                                         sycl::range<3>(1, 1, 1)),
                       [=](sycl::nd_item<3> item_ct1) {
                           kernel_hset_pointer(output_array, input, lda, row,
                                               column, batch_offset, item_ct1);
                       });
}
