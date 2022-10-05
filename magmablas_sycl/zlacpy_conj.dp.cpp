/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define BLOCK_SIZE 64

/******************************************************************************/
// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old zswap routine. Update to new zlacpy code for 2D matrix?

void zlacpy_conj_kernel(
    int n,
    magmaDoubleComplex *A1, int lda1,
    magmaDoubleComplex *A2, int lda2 , sycl::nd_item<3> item_ct1)
{
    int x = item_ct1.get_local_id(2) +
            item_ct1.get_local_range(2) * item_ct1.get_group(2);
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_Z_CONJ( A1[offset1] );
    }
}


/******************************************************************************/
extern "C" void 
magmablas_zlacpy_conj(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, magma_int_t lda1, 
    magmaDoubleComplex_ptr dA2, magma_int_t lda2,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, BLOCK_SIZE);
    sycl::range<3> blocks(1, 1, magma_ceildiv(n, BLOCK_SIZE));
    /*
    DPCT1049:1079: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zlacpy_conj_kernel(n, dA1, lda1, dA2, lda2,
                                              item_ct1);
                       });
}
