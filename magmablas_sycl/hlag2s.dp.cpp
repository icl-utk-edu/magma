/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
    // for CUDA_VERSION
#include "magma_internal.h"

#if defined(MAGMA_HAVE_HIP)
#include <hip/hip_fp16.h>
#endif

#define BLK_X 32
#define BLK_Y 4

/******************************************************************************/

void hlag2s_device(
    int m, int n,
    magmaHalf_const_ptr A, int lda,
    float             *SA, int ldsa , sycl::nd_item<3> item_ct1)
{
    const int gtx = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    const int gty = item_ct1.get_group(1) * BLK_Y + item_ct1.get_local_id(1);

    for (int j = gty; j < n; j += item_ct1.get_group_range(1) * BLK_Y) {
        for (int i = gtx; i < m; i += item_ct1.get_group_range(2) * BLK_X) {
            SA[j * ldsa + i] =
                sycl::vec<sycl::half, 1>{A[j * lda + i]}
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        }
    }
}

/******************************************************************************/

void hlag2s_kernel(
        int m, int n,
        magmaHalf_const_ptr dA, int lda,
        float             *dSA, int ldsa , sycl::nd_item<3> item_ct1)
{
    hlag2s_device(m, n, dA, lda, dSA, ldsa, item_ct1);
}

/******************************************************************************/

void hlag2s_kernel_batched(
        int m, int n,
        magmaHalf const * const * dAarray, int lda,
        float**                  dSAarray, int ldsa , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    hlag2s_device(m, n, dAarray[batchid], lda, dSAarray[batchid], ldsa,
                  item_ct1);
}

/******************************************************************************/
extern "C" void
magmablas_hlag2s(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA, magma_int_t lda,
    float             *dSA, magma_int_t ldsa,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < max(1,m) )
        info = -4;
    else if ( ldsa < max(1,m) )
        info = -6;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    const int max_gridy = 65000; // the kernel can work with any gridx/gridy dimension
    sycl::range<3> threads(1, BLK_Y, BLK_X);
    sycl::range<3> grid(1, min(max_gridy, magma_ceildiv(n, BLK_Y)),
                        magma_ceildiv(m, BLK_X));
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           hlag2s_kernel(m, n, dA, lda, dSA, ldsa, item_ct1);
                       });
}


/******************************************************************************/
extern "C" void
magmablas_hlag2s_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf const * const * dAarray, magma_int_t lda,
    float               **dSAarray, magma_int_t ldsa,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < max(1,m) )
        info = -4;
    else if ( ldsa < max(1,m) )
        info = -6;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    sycl::range<3> threads(1, BLK_Y, BLK_X);
    const int maxBatch = queue->get_maxBatch();
    for(int i = 0; i < batchCount; i+=maxBatch){
        magma_int_t batch = min(maxBatch, batchCount-i);
        sycl::range<3> grid(batch, magma_ceildiv(n, BLK_Y),
                            magma_ceildiv(m, BLK_X));
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               hlag2s_kernel_batched(m, n, dAarray + i, lda,
                                                     dSAarray + i, ldsa,
                                                     item_ct1);
                           });
    }
}
