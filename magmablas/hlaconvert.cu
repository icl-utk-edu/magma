/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       
       @precisions magmaHalf

*/
#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"

#if CUDA_VERSION >= 7500

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
void convert_dp2hp_device(
    int m, int n,
    const double *dA, int ldda,
          magmaHalf *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = __float2half( float(dA[j*ldda]) );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = __float2half( float(dA[j*ldda]) );
            }
        }
    }
}
/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to zlaset, zlacpy, zlag2c, clag2z, zgeadd.
*/
static __device__
void convert_hp2dp_device(
    int m, int n,
    const magmaHalf *dA, int ldda,
          double *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = double(__half2float( dA[j*ldda] ));
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = double(__half2float( dA[j*ldda] ));
            }
        }
    }
}
/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to zlaset, zlacpy, zlag2c, clag2z, zgeadd.
*/
static __device__
void convert_sp2hp_device(
    int m, int n,
    const float  *dA, int ldda,
          magmaHalf *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = __float2half( dA[j*ldda] );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = __float2half( dA[j*ldda] );
            }
        }
    }
}
/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to zlaset, zlacpy, zlag2c, clag2z, zgeadd.
*/
static __device__
void convert_hp2sp_device(
    int m, int n,
    const magmaHalf *dA, int ldda,
          float  *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = __half2float( dA[j*ldda] );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = __half2float( dA[j*ldda] );
            }
        }
    }
}

/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/
__global__
void convert_dp2hp_kernel(
    int m, int n,
    const double *dA, int ldda,
          magmaHalf *dB, int lddb )
{
#if CUDA_VERSION >= 7500
    convert_dp2hp_device(m, n, dA, ldda, dB, lddb);
#endif
}
/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/
__global__
void convert_hp2dp_kernel(
    int m, int n,
    const magmaHalf *dA, int ldda,
          double *dB, int lddb )
{
#if CUDA_VERSION >= 7500
    convert_hp2dp_device(m, n, dA, ldda, dB, lddb);
#endif
}

/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/
__global__
void convert_sp2hp_kernel(
    int m, int n,
    const float  *dA, int ldda,
          magmaHalf *dB, int lddb )
{
#if CUDA_VERSION >= 7500
    convert_sp2hp_device(m, n, dA, ldda, dB, lddb);
#endif
}
/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/
__global__
void convert_hp2sp_kernel(
    int m, int n,
    const magmaHalf *dA, int ldda,
          float  *dB, int lddb )
{
#if CUDA_VERSION >= 7500
    convert_hp2sp_device(m, n, dA, ldda, dB, lddb);
#endif
}

/***************************************************************************//**
    Purpose
    -------
    HLACONVERT convert all or part of a two-dimensional matrix dA to another
    matrix dB.
    
    Arguments
    ---------
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[out]
    dB      COMPLEX_16 array, dimension (LDDB,N)
            The M-by-N matrix dB.
            On exit, dB = dA in the locations specified by UPLO.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_laconvert
*******************************************************************************/
extern "C" void
magmablas_convert_sp2hp(
    magma_int_t m, magma_int_t n,
    const float *dA, magma_int_t ldda,
         magmaHalf *dB, magma_int_t lddb,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;
    else if ( lddb < max(1,m))
        info = -6;
    
    if ( info != 0 ) {
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
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = magma_ceildiv( mm, BLK_X );
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = magma_ceildiv( nn, BLK_Y );
            convert_sp2hp_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }
}
/***************************************************************************/
extern "C" void
magmablas_convert_hp2sp(
    magma_int_t m, magma_int_t n,
    const magmaHalf *dA, magma_int_t ldda,
          float  *dB, magma_int_t lddb,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;
    else if ( lddb < max(1,m))
        info = -6;
    
    if ( info != 0 ) {
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
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = magma_ceildiv( mm, BLK_X );
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = magma_ceildiv( nn, BLK_Y );
            convert_hp2sp_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }

}
/***************************************************************************/
extern "C" void
magmablas_convert_dp2hp(
    magma_int_t m, magma_int_t n,
    const double *dA, magma_int_t ldda,
          magmaHalf *dB, magma_int_t lddb,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;
    else if ( lddb < max(1,m))
        info = -6;
    
    if ( info != 0 ) {
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
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = magma_ceildiv( mm, BLK_X );
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = magma_ceildiv( nn, BLK_Y );
            convert_dp2hp_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }
}
/***************************************************************************/

extern "C" void
magmablas_convert_hp2dp(
    magma_int_t m, magma_int_t n,
    const magmaHalf *dA, magma_int_t ldda,
          double *dB, magma_int_t lddb,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;
    else if ( lddb < max(1,m))
        info = -6;
    
    if ( info != 0 ) {
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
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = magma_ceildiv( mm, BLK_X );
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = magma_ceildiv( nn, BLK_Y );
            convert_hp2dp_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }

}

#endif
