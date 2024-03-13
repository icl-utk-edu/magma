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
#include "magma_templates.h"

// size of work for a thread block
#define BLK_M 16
#define BLK_N 16

// BLK_K gets defined in magmablas_zgemm_reduce,
// because it depends on the CUDA architecture at runtime.


/******************************************************************************/
// BLK_K size is templated, as it depends on CUDA architecture at runtime.
// Hmm... how to compile for both CUDA arch 1.x and 2.x?

template< int BLK_K >

void zgemm_reduce_kernel(
    int m, int n, int k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* __restrict__ dA, int lda,
    const magmaDoubleComplex* __restrict__ dB, int ldb,
    magmaDoubleComplex beta,
    magmaDoubleComplex      * __restrict__ dC, int ldc,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 3> sum)
{
    const int tx = item_ct1.get_local_id(2);

    if (item_ct1.get_group(2) * BLK_M + item_ct1.get_local_id(1) < m &&
        item_ct1.get_group(1) * BLK_N + item_ct1.get_local_id(0) < n) {
        dA += (item_ct1.get_group(2) * BLK_M + item_ct1.get_local_id(1)) * lda;
        dB += (item_ct1.get_group(1) * BLK_N + item_ct1.get_local_id(0)) * ldb;
        dC +=
            item_ct1.get_group(2) * BLK_M + item_ct1.get_group(1) * BLK_N * ldc;

        // was: sum[BLK_M][BLK_N+1][BLK_K+1];
        // moved 3rd dimension to 1st dimension to make magma_sum_reduce_3d interface nicer.

        magmaDoubleComplex lsum;
        
        /*  w := v**H * C  */
        lsum = MAGMA_Z_ZERO;
        for( int j = tx; j < k; j += BLK_K )
            lsum += MAGMA_Z_CONJ( dA[j] )* dB[j];

        sum[tx][item_ct1.get_local_id(1)][item_ct1.get_local_id(0)] = lsum;
        magma_sum_reduce_3d<BLK_K, BLK_M + 1, BLK_N + 1>(
            tx, item_ct1.get_local_id(1), item_ct1.get_local_id(0), sum,
            item_ct1);

        /*  C := C - v * w  */
        /*
        DPCT1065:358: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            if (MAGMA_Z_EQUAL(beta, MAGMA_Z_ZERO))
                dC[item_ct1.get_local_id(1) + item_ct1.get_local_id(0) * ldc] =
                    alpha *
                    sum[0][item_ct1.get_local_id(1)][item_ct1.get_local_id(0)];
            else
                dC[item_ct1.get_local_id(1) + item_ct1.get_local_id(0) * ldc] =
                    beta * dC[item_ct1.get_local_id(1) +
                              item_ct1.get_local_id(0) * ldc] +
                    alpha * sum[0][item_ct1.get_local_id(1)]
                               [item_ct1.get_local_id(0)];
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEMM_REDUCE  performs one of the matrix-matrix operations
    
        C := alpha*A^T*B + beta*C,
    
    where alpha and beta are scalars, and A, B and C are matrices, with A
    a k-by-m matrix, B a k-by-n matrix, and C an m-by-n matrix.
    
    This routine is tuned for m, n << k. Typically, m and n are expected
    to be less than 128.

    @ingroup magma_gemm
*******************************************************************************/
extern "C" void
magmablas_zgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( k < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( lddb < k )
        info = -8;
    else if ( lddc < m )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    const int NUM_THREADS = 1024;
    const int BLK_K = (NUM_THREADS / (BLK_M * BLK_N)); // == 4
    sycl::range<3> threads(BLK_N, BLK_M, BLK_K);
    sycl::range<3> blocks(1, magma_ceildiv(n, BLK_N),
                          magma_ceildiv(m, BLK_M));
    /*
    DPCT1049:360: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 3>
                sum_acc_ct1(
                    sycl::range<3>(BLK_K, BLK_M+1, BLK_N+1),
                    cgh);

            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 zgemm_reduce_kernel<BLK_K>(
                                     m, n, k, alpha, dA, ldda, dB, lddb,
                                     beta, dC, lddc, item_ct1, sum_acc_ct1);
                             });
        });
}
