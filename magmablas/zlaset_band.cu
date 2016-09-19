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
__global__
void zlaset_band_upper(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda)
{
    int k   = blockDim.x;
    int ibx = blockIdx.x * NB;
    int ind = ibx + threadIdx.x - k + 1;
    
    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (threadIdx.x == k-1)
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
__global__
void zlaset_band_lower(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda)
{
    //int k   = blockDim.x;
    int ibx = blockIdx.x * NB;
    int ind = ibx + threadIdx.x;
    
    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (threadIdx.x == 0)
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
        dim3 threads( min(k,n) );
        dim3 grid( magma_ceildiv( min(m+k-1,n), NB ) );
        zlaset_band_upper<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dA, ldda);
    }
    else if (uplo == MagmaLower) {
        dim3 threads( min(k,m) );
        dim3 grid( magma_ceildiv( min(m,n), NB ) );
        zlaset_band_lower<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dA, ldda);
    }
}
