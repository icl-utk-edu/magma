/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Mark Gates
       
       @precisions normal z -> s d c

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 64

/******************************************************************************/
/*
    GPU kernel for setting the k-1 super-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil((m+k-1)/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread k-1 does the main diagonal, thread k-2 the first super-diagonal, etc.

      block 0           block 1
      0                           => skip above matrix
      1 0                         => skip above matrix
      2 1 0                       => skip above matrix
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                 | 3 2 1 0 ]
    [                 |   3 2 1 ]
                      |     3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=10, n=12, k=4, nb=8. Each column is done in parallel.
*/

void zlaset_band_upper(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda, sycl::nd_item<3> item_ct1)
{
    int k = item_ct1.get_local_range(2);
    int ibx = item_ct1.get_group(2) * NB;
    int ind = ibx + item_ct1.get_local_id(2) - k + 1;

    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (item_ct1.get_local_id(2) == k - 1)
        value = diag;

    #pragma unroll
    for (int j=0; j < NB; j++) {
        if (ibx + j < n && ind + j >= 0 && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}


/******************************************************************************/
/*
    GPU kernel for setting the k-1 sub-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil(m/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread 0 does the main diagonal, thread 1 the first sub-diagonal, etc.
    
      block 0           block 1
    [ 0               |         ]
    [ 1 0             |         ]
    [ 2 1 0           |         ]
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                   3 2 1 0 ]
    [                     3 2 1 ]
                            3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=13, n=12, k=4, nb=8. Each column is done in parallel.
*/

void zlaset_band_lower(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda, sycl::nd_item<3> item_ct1)
{
    //int k   = blockDim.x;
    int ibx = item_ct1.get_group(2) * NB;
    int ind = ibx + item_ct1.get_local_id(2);

    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (item_ct1.get_local_id(2) == 0)
        value = diag;

    #pragma unroll
    for (int j=0; j < NB; j++) {
        if (ibx + j < n && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLASET_BAND initializes the main diagonal of dA to DIAG,
    and the K-1 sub- or super-diagonals to OFFDIAG.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    k       INTEGER
            The number of diagonals to set, including the main diagonal.  K >= 0.
            Currently, K <= 1024 due to CUDA restrictions (max. number of threads per block).
    
    @param[in]
    offdiag COMPLEX_16
            Off-diagonal elements in the band are set to OFFDIAG.
    
    @param[in]
    diag    COMPLEX_16
            All the main diagonal elements are set to DIAG.
    
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = ALPHA, 1 <= i <= m, 1 <= j <= n where i != j, abs(i-j) < k;
            and      A(i,i) = BETA,  1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Stream to execute ZLASET in.
    
    @ingroup magma_laset_band
*******************************************************************************/
extern "C" void
magmablas_zlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 || k > 1024 )
        info = -4;
    else if ( ldda < max(1,m) )
        info = -6;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if (uplo == MagmaUpper) {
        sycl::range<3> threads(1, 1, min(k, n));
        sycl::range<3> grid(1, 1, magma_ceildiv(min(m + k - 1, n), NB));
        /*
        DPCT1049:1243: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlaset_band_upper(m, n, offdiag, diag, dA, ldda,
                                                 item_ct1);
                           });
    }
    else if (uplo == MagmaLower) {
        sycl::range<3> threads(1, 1, min(k, m));
        sycl::range<3> grid(1, 1, magma_ceildiv(min(m, n), NB));
        /*
        DPCT1049:1244: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlaset_band_lower(m, n, offdiag, diag, dA, ldda,
                                                 item_ct1);
                           });
    }
}
