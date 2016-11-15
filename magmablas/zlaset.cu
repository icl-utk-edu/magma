/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @precisions normal z -> s d c

*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

// To deal with really large matrices, this launchs multiple super blocks,
// each with up to 64K-1 x 64K-1 thread blocks, which is up to 4194240 x 4194240 matrix with BLK=64.
// CUDA architecture 2.0 limits each grid dimension to 64K-1.
// Instances arose for vectors used by sparse matrices with M > 4194240, though N is small.
const magma_int_t max_blocks = 65535;

// BLK_X and BLK_Y need to be equal for zlaset_q to deal with diag & offdiag
// when looping over super blocks.
// Formerly, BLK_X and BLK_Y could be different.
#define BLK_X 64
#define BLK_Y BLK_X

/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to zlaset, zlacpy, zlag2c, clag2z, zgeadd.
*/
static __device__
void zlaset_full_device(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag || above diag || offdiag == diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y || ind + BLK_X <= iby || MAGMA_Z_EQUAL( offdiag, diag )));
    /* do only rows inside matrix */
    if ( ind < m ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block or offdiag == diag
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else
                    A[j*lda] = offdiag;
            }
        }
    }
}


/******************************************************************************/
/*
    Similar to zlaset_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to zlaset, zlacpy, zlat2c, clat2z.
*/
static __device__
void zlaset_lower_device(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind > iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


/******************************************************************************/
/*
    Similar to zlaset_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to zlaset, zlacpy, zlat2c, clat2z.
*/
static __device__
void zlaset_upper_device(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind < iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/
__global__
void zlaset_full_kernel(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, int ldda )
{
    zlaset_full_device(m, n, offdiag, diag, dA, ldda);
}

__global__
void zlaset_lower_kernel(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, int ldda )
{
    zlaset_lower_device(m, n, offdiag, diag, dA, ldda);
}

__global__
void zlaset_upper_kernel(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, int ldda )
{
    zlaset_upper_device(m, n, offdiag, diag, dA, ldda);
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions for the batched routine.
*/
__global__
void zlaset_full_kernel_batched(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, int ldda )
{
    int batchid = blockIdx.z;
    zlaset_full_device(m, n, offdiag, diag, dAarray[batchid], ldda);
}

__global__
void zlaset_lower_kernel_batched(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, int ldda )
{
    int batchid = blockIdx.z;
    zlaset_lower_device(m, n, offdiag, diag, dAarray[batchid], ldda);
}

__global__
void zlaset_upper_kernel_batched(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, int ldda )
{
    int batchid = blockIdx.z;
    zlaset_upper_device(m, n, offdiag, diag, dAarray[batchid], ldda);
}
/******************************************************************************/
/*
    kernel wrappers to call the device functions for the vbatched routine.
*/
__global__
void zlaset_full_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, magma_int_t* ldda )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if( blockIdx.x >= (my_m+BLK_X-1)/BLK_X ) return;
    if( blockIdx.y >= (my_n+BLK_Y-1)/BLK_Y ) return;
    zlaset_full_device(my_m, my_n, offdiag, diag, dAarray[batchid], (int)ldda[batchid]);
}

__global__
void zlaset_lower_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, magma_int_t* ldda )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if( blockIdx.x >= (my_m+BLK_X-1)/BLK_X ) return;
    if( blockIdx.y >= (my_n+BLK_Y-1)/BLK_Y ) return;
    zlaset_lower_device(my_m, my_n, offdiag, diag, dAarray[batchid], (int)ldda[batchid]);
}

__global__
void zlaset_upper_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, magma_int_t* ldda )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if( blockIdx.x >= (my_m+BLK_X-1)/BLK_X ) return;
    if( blockIdx.y >= (my_n+BLK_Y-1)/BLK_Y ) return;
    zlaset_upper_device(my_m, my_n, offdiag, diag, dAarray[batchid], (int)ldda[batchid]);
}


/***************************************************************************//**
    Purpose
    -------
    ZLASET initializes a 2-D array A to DIAG on the diagonal and
    OFFDIAG on the off-diagonals.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
      -     = MagmaFull:       All of the matrix dA
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    offdiag COMPLEX_16
            The scalar OFFDIAG. (In LAPACK this is called ALPHA.)
    
    @param[in]
    diag    COMPLEX_16
            The scalar DIAG. (In LAPACK this is called BETA.)
    
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = OFFDIAG, 1 <= i <= m, 1 <= j <= n, i != j;
            and      A(i,i) = DIAG,    1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_laset
*******************************************************************************/
extern "C"
void magmablas_zlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue)
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m) )
        info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    assert( BLK_X == BLK_Y );
    const magma_int_t super_NB = max_blocks*BLK_X;
    dim3 super_grid( magma_ceildiv( m, super_NB ), magma_ceildiv( n, super_NB ) );
    
    dim3 threads( BLK_X, 1 );
    dim3 grid;
    
    magma_int_t mm, nn;
    if (uplo == MagmaLower) {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = magma_ceildiv( mm, BLK_X );
            for( unsigned int j=0; j < super_grid.y && j <= i; ++j ) {  // from left to diagonal
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = magma_ceildiv( nn, BLK_Y );
                if ( i == j ) {  // diagonal super block
                    zlaset_lower_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, offdiag, diag, dA(i*super_NB, j*super_NB), ldda );
                }
                else {           // off diagonal super block
                    zlaset_full_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, offdiag, offdiag, dA(i*super_NB, j*super_NB), ldda );
                }
            }
        }
    }
    else if (uplo == MagmaUpper) {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = magma_ceildiv( mm, BLK_X );
            for( unsigned int j=i; j < super_grid.y; ++j ) {  // from diagonal to right
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = magma_ceildiv( nn, BLK_Y );
                if ( i == j ) {  // diagonal super block
                    zlaset_upper_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, offdiag, diag, dA(i*super_NB, j*super_NB), ldda );
                }
                else {           // off diagonal super block
                    zlaset_full_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, offdiag, offdiag, dA(i*super_NB, j*super_NB), ldda );
                }
            }
        }
    }
    else {
        // if continuous in memory & set to zero, cudaMemset is faster.
        // TODO: use cudaMemset2D ?
        if ( m == ldda &&
             MAGMA_Z_EQUAL( offdiag, MAGMA_Z_ZERO ) &&
             MAGMA_Z_EQUAL( diag,    MAGMA_Z_ZERO ) )
        {
            size_t size = m*n;
            cudaError_t err = cudaMemsetAsync( dA, 0, size*sizeof(magmaDoubleComplex), queue->cuda_stream() );
            assert( err == cudaSuccess );
            MAGMA_UNUSED( err );
        }
        else {
            for( unsigned int i=0; i < super_grid.x; ++i ) {
                mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
                grid.x = magma_ceildiv( mm, BLK_X );
                for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                    nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                    grid.y = magma_ceildiv( nn, BLK_Y );
                    if ( i == j ) {  // diagonal super block
                        zlaset_full_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                            ( mm, nn, offdiag, diag, dA(i*super_NB, j*super_NB), ldda );
                    }
                    else {           // off diagonal super block
                        zlaset_full_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                            ( mm, nn, offdiag, offdiag, dA(i*super_NB, j*super_NB), ldda );
                    }
                }
            }
        }
    }
}


/******************************************************************************/
extern "C"
void magmablas_zlaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m) )
        info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1, 1 );
    dim3 grid( magma_ceildiv( m, BLK_X ), magma_ceildiv( n, BLK_Y ), batchCount );
    
    if (uplo == MagmaLower) {
        zlaset_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dAarray, ldda);
    }
    else if (uplo == MagmaUpper) {
        zlaset_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dAarray, ldda);
    }
    else {
        zlaset_full_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dAarray, ldda);
    }
}
/******************************************************************************/
extern "C"
void magmablas_zlaset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( max_m < 0 )
        info = -2;
    else if ( max_n < 0 )
        info = -3;
    //else if ( ldda < max(1,m) )
    //    info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( max_m == 0 || max_n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1, 1 );
    dim3 grid( magma_ceildiv( max_m, BLK_X ), magma_ceildiv( max_n, BLK_Y ), batchCount );
    
    if (uplo == MagmaLower) {
        zlaset_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dAarray, ldda);
    }
    else if (uplo == MagmaUpper) {
        zlaset_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dAarray, ldda);
    }
    else {
        zlaset_full_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>> (m, n, offdiag, diag, dAarray, ldda);
    }
}
