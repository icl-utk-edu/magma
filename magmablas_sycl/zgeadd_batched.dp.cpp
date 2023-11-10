/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 64

/******************************************************************************/
/*
    Batches zlacpy of multiple arrays;
    y-dimension of grid is different arrays,
    x-dimension of grid is blocks for each array.
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread adds one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.

    TODO. Block in both directions, for large matrices.
    E.g., each block does 64x64 tile, instead of 64xN tile.
*/
void
zgeadd_batched_kernel(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex * const *dAarray, int ldda,
    magmaDoubleComplex              **dBarray, int lddb ,
    sycl::nd_item<3> item_ct1)
{
    // dA and dB iterate across row i
    const magmaDoubleComplex *dA = dAarray[item_ct1.get_group(1)];
    magmaDoubleComplex *dB = dBarray[item_ct1.get_group(1)];
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if ( i < m ) {
        dA += i;
        dB += i;
        const magmaDoubleComplex *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = alpha*(*dA) + (*dB);
            dA += ldda;
            dB += lddb;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEADD adds two sets of matrices, dAarray[i] = alpha*dAarray[i] + dBarray[i],
    for i = 0, ..., batchCount-1.

    Arguments
    ---------

    @param[in]
    m       INTEGER
            The number of rows of each matrix dAarray[i].  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix dAarray[i].  N >= 0.

    @param[in]
    alpha   COMPLEX_16
            The scalar alpha.

    @param[in]
    dAarray array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a COMPLEX_16 array, dimension (LDDA,N)
            The m by n matrices dAarray[i].

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dAarray[i].  LDDA >= max(1,M).

    @param[in,out]
    dBarray array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a COMPLEX_16 array, dimension (LDDB,N)
            The m by n matrices dBarray[i].

    @param[in]
    lddb    INTEGER
            The leading dimension of each array dBarray[i].  LDDB >= max(1,M).

    @param[in]
    batchCount INTEGER
            The number of matrices to add; length of dAarray and dBarray.
            batchCount >= 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_geadd_batched
*******************************************************************************/
extern "C" void
magmablas_zgeadd_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    else if ( batchCount < 0 )
        info = -8;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( m == 0 || n == 0 || batchCount == 0 )
        return;

    sycl::range<3> threads(1, 1, NB);
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(1, ibatch, magma_ceildiv(m, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zgeadd_batched_kernel(m, n, alpha, dAarray + i,
                                                     ldda, dBarray + i, lddb,
                                                     item_ct1);
                           });
    }
}
