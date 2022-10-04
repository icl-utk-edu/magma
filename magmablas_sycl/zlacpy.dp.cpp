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
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

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
static 
void zlacpy_full_device(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
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
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/******************************************************************************/
/*
    Similar to zlacpy_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to zlaset, zlacpy, zlat2c, clat2z.
*/
static 
void zlacpy_lower_device(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/******************************************************************************/
/*
    Similar to zlacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to zlaset, zlacpy, zlat2c, clat2z.
*/
static 
void zlacpy_upper_device(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    dB[j*lddb] = dA[j*ldda];
                }
            }
        }
    }
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/

void zlacpy_full_kernel(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    zlacpy_full_device(m, n, dA, ldda, dB, lddb, item_ct1);
}


void zlacpy_lower_kernel(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    zlacpy_lower_device(m, n, dA, ldda, dB, lddb, item_ct1);
}


void zlacpy_upper_kernel(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    zlacpy_upper_device(m, n, dA, ldda, dB, lddb, item_ct1);
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions for the batched routine.
*/

void zlacpy_full_kernel_batched(
    int m, int n,
    magmaDoubleComplex const * const *dAarray, magma_int_t Ai, magma_int_t Aj, int ldda,
    magmaDoubleComplex **dBarray, magma_int_t Bi, magma_int_t Bj, int lddb ,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    zlacpy_full_device(m, n, dAarray[batchid] + Aj * ldda + Ai, ldda,
                       dBarray[batchid] + Bj * lddb + Bi, lddb, item_ct1);
}


void zlacpy_lower_kernel_batched(
    int m, int n,
    magmaDoubleComplex const * const *dAarray, magma_int_t Ai, magma_int_t Aj, int ldda,
    magmaDoubleComplex **dBarray, magma_int_t Bi, magma_int_t Bj, int lddb ,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    zlacpy_lower_device(m, n, dAarray[batchid] + Aj * ldda + Ai, ldda,
                        dBarray[batchid] + Bj * lddb + Bi, lddb, item_ct1);
}


void zlacpy_upper_kernel_batched(
    int m, int n,
    magmaDoubleComplex const * const *dAarray, magma_int_t Ai, magma_int_t Aj, int ldda,
    magmaDoubleComplex **dBarray, magma_int_t Bi, magma_int_t Bj, int lddb ,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    zlacpy_upper_device(m, n, dAarray[batchid] + Aj * ldda + Ai, ldda,
                        dBarray[batchid] + Bj * lddb + Bi, lddb, item_ct1);
}

/******************************************************************************/
/*
    kernel wrappers to call the device functions for the vbatched routine.
*/

void zlacpy_full_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex const * const *dAarray, magma_int_t* ldda,
    magmaDoubleComplex **dBarray, magma_int_t* lddb , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_m, BLK_X)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_n, BLK_Y)) return;

    zlacpy_full_device(my_m, my_n, dAarray[batchid], (int)ldda[batchid],
                       dBarray[batchid], (int)lddb[batchid], item_ct1);
}


void zlacpy_lower_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex const * const *dAarray, magma_int_t* ldda,
    magmaDoubleComplex **dBarray, magma_int_t* lddb , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_m, BLK_X)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_n, BLK_Y)) return;

    zlacpy_lower_device(my_m, my_n, dAarray[batchid], (int)ldda[batchid],
                        dBarray[batchid], (int)lddb[batchid], item_ct1);
}


void zlacpy_upper_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex const * const *dAarray, magma_int_t* ldda,
    magmaDoubleComplex **dBarray, magma_int_t* lddb , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_m, BLK_X)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_n, BLK_Y)) return;

    zlacpy_upper_device(my_m, my_n, dAarray[batchid], (int)ldda[batchid],
                        dBarray[batchid], (int)lddb[batchid], item_ct1);
}


/***************************************************************************//**
    Purpose
    -------
    ZLACPY copies all or part of a two-dimensional matrix dA to another
    matrix dB.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be copied to dB.
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

    @ingroup magma_lacpy
*******************************************************************************/
extern "C" void
magmablas_zlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)

    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( m == 0 || n == 0 ) {
        return;
    }

    assert( BLK_X == BLK_Y );
    const magma_int_t super_NB = max_blocks*BLK_X;
    sycl::range<3> super_grid(1, magma_ceildiv(n, super_NB),
                              magma_ceildiv(m, super_NB));

    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, 1);

    magma_int_t mm, nn;
    if ( uplo == MagmaLower ) {
        for (unsigned int i = 0; i < super_grid[2]; ++i) {
            mm = (i == super_grid[2] - 1 ? m % super_NB : super_NB);
            grid[2] = magma_ceildiv(mm, BLK_X);
            for (unsigned int j = 0; j < super_grid[1] && j <= i;
                 ++j) { // from left to diagonal
                nn = (j == super_grid[1] - 1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv(nn, BLK_Y);
                if ( i == j ) {  // diagonal super block
                    /*
                    DPCT1049:1080: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(queue->cuda_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlacpy_lower_kernel(
                                    mm, nn, dA(i * super_NB, j * super_NB),
                                    ldda, dB(i * super_NB, j * super_NB), lddb,
                                    item_ct1);
                            });
                }
                else {           // off diagonal super block
                    /*
                    DPCT1049:1081: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(queue->cuda_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlacpy_full_kernel(
                                    mm, nn, dA(i * super_NB, j * super_NB),
                                    ldda, dB(i * super_NB, j * super_NB), lddb,
                                    item_ct1);
                            });
                }
            }
        }
    }
    else if ( uplo == MagmaUpper ) {
        for (unsigned int i = 0; i < super_grid[2]; ++i) {
            mm = (i == super_grid[2] - 1 ? m % super_NB : super_NB);
            grid[2] = magma_ceildiv(mm, BLK_X);
            for (unsigned int j = i; j < super_grid[1];
                 ++j) { // from diagonal to right
                nn = (j == super_grid[1] - 1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv(nn, BLK_Y);
                if ( i == j ) {  // diagonal super block
                    /*
                    DPCT1049:1082: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(queue->cuda_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlacpy_upper_kernel(
                                    mm, nn, dA(i * super_NB, j * super_NB),
                                    ldda, dB(i * super_NB, j * super_NB), lddb,
                                    item_ct1);
                            });
                }
                else {           // off diagonal super block
                    /*
                    DPCT1049:1083: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(queue->cuda_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlacpy_full_kernel(
                                    mm, nn, dA(i * super_NB, j * super_NB),
                                    ldda, dB(i * super_NB, j * super_NB), lddb,
                                    item_ct1);
                            });
                }
            }
        }
    }
    else {
        // TODO: use cudaMemcpy or cudaMemcpy2D ?
        for (unsigned int i = 0; i < super_grid[2]; ++i) {
            mm = (i == super_grid[2] - 1 ? m % super_NB : super_NB);
            grid[2] = magma_ceildiv(mm, BLK_X);
            for (unsigned int j = 0; j < super_grid[1]; ++j) { // full row
                nn = (j == super_grid[1] - 1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv(nn, BLK_Y);
                /*
                DPCT1049:1084: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->cuda_stream()))
                    ->parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zlacpy_full_kernel(
                                mm, nn, dA(i * super_NB, j * super_NB), ldda,
                                dB(i * super_NB, j * super_NB), lddb, item_ct1);
                        });
            }
        }
    }
}

extern "C" void
magmablas_zlacpy_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex_ptr             dBarray[], magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, BLK_X);
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, magma_ceildiv(n, BLK_Y),
                            magma_ceildiv(m, BLK_X));

        if ( uplo == MagmaLower ) {
            /*
            DPCT1049:1085: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlacpy_lower_kernel_batched(
                                       m, n, dAarray + i, Ai, Aj, ldda,
                                       dBarray + i, Bi, Bj, lddb, item_ct1);
                               });
        }
        else if ( uplo == MagmaUpper ) {
            /*
            DPCT1049:1086: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlacpy_upper_kernel_batched(
                                       m, n, dAarray + i, Ai, Aj, ldda,
                                       dBarray + i, Bi, Bj, lddb, item_ct1);
                               });
        }
        else {
            /*
            DPCT1049:1087: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlacpy_full_kernel_batched(
                                       m, n, dAarray + i, Ai, Aj, ldda,
                                       dBarray + i, Bi, Bj, lddb, item_ct1);
                               });
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    ZLACPY_BATCHED copies all or part of each two-dimensional matrix
    dAarray[i] to matrix dBarray[i], for 0 <= i < batchcount.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies the part of each matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dA

    @param[in]
    m       INTEGER
            The number of rows of each matrix dA.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix dA.  N >= 0.

    @param[in]
    dAarray COMPLEX_16* array, dimension (batchCount)
            Array of pointers to the matrices dA, where each dA is of dimension (LDDA,N).
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dA.  LDDA >= max(1,M).

    @param[out]
    dBarray COMPLEX_16* array, dimension (batchCount)
            Array of pointers to the matrices dB, where each dB is of dimension (LDDB,N).
            The M-by-N matrix dB.
            On exit, dB = dA in the locations specified by UPLO.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array dB.  LDDB >= max(1,M).

    @param[in]
    batchCount  Number of matrices in dAarray and dBarray.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacpy_batched
*******************************************************************************/
extern "C" void
magmablas_zlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr             dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
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

    if ( m == 0 || n == 0 || batchCount == 0 ) {
        return;
    }

    magmablas_zlacpy_internal_batched(
    uplo, m, n,
    dAarray, 0, 0, ldda,
    dBarray, 0, 0, lddb,
    batchCount, queue );
}

/***************************************************************************//**
    Purpose
    -------
    ZLACPY_VBATCHED copies all or part of each two-dimensional matrix
    dAarray[i] to matrix dBarray[i], for 0 <= i < batchcount.
    Matrices are assumed to generally have different sizes/leading dimensions

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies the part of each matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dA

    @param[in]
    max_m   INTEGER.
            maximum value of m.

    @param[in]
    max_n   INTEGER.
            maximum value of n.

    @param[in]
    m       INTEGER array, dimension (batchCount).
            Each is the number of rows of each matrix dA.  M >= 0.

    @param[in]
    n       INTEGER array, dimension (batchCount).
            The number of columns of each matrix dA.  N >= 0.

    @param[in]
    dAarray Array of pointers , dimension (batchCount)
            Each is a COMPLEX_16 array dA, where the ith matrix dA is of dimension (ldda[i],n[i]).
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.

    @param[in]
    ldda    INTEGER array, dimension (batchCount).
            Each is the leading dimension of each array dA. For the ith matrix dA ldda[i] >= max(1,m[i]).

    @param[out]
    dBarray Array of pointers, dimension(batchCount).
            Each is a COMPLEX_16 array dB, where the ith matrix dB is of dimension (lddb[i],n[i]).
            The M-by-N matrix dB.
            On exit, dB = dA in the locations specified by UPLO.

    @param[in]
    lddb    INTEGER array, dimension (batchCount).
            Each is the leading dimension of each array dB. For the ith matrix dB lddb[i] >= max(1,m[i]).

    @param[in]
    batchCount  Number of matrices in dAarray and dBarray.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacpy_batched
*******************************************************************************/
extern "C" void
magmablas_zlacpy_vbatched(
    magma_uplo_t uplo,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex const * const * dAarray, magma_int_t* ldda,
    magmaDoubleComplex**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    //else if ( m < 0 )
    //    info = -2;
    //else if ( n < 0 )
    //    info = -3;
    //else if ( ldda < max(1,m))
    //    info = -5;
    //else if ( lddb < max(1,m))
    //    info = -7;
    else if ( batchCount < 0 )
        info = -8;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( max_m == 0 || max_n == 0 || batchCount == 0 ) {
        return;
    }

    sycl::range<3> threads(1, 1, BLK_X);

    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, magma_ceildiv(max_n, BLK_Y),
                            magma_ceildiv(max_m, BLK_X));

        if ( uplo == MagmaLower ) {
            /*
            DPCT1049:1088: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlacpy_lower_kernel_vbatched(
                                       m + i, n + i, dAarray + i, ldda + i,
                                       dBarray + i, lddb + i, item_ct1);
                               });
        }
        else if ( uplo == MagmaUpper ) {
            /*
            DPCT1049:1089: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlacpy_upper_kernel_vbatched(
                                       m + i, n + i, dAarray + i, ldda + i,
                                       dBarray + i, lddb + i, item_ct1);
                               });
        }
        else {
            /*
            DPCT1049:1090: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlacpy_full_kernel_vbatched(
                                       m + i, n + i, dAarray + i, ldda + i,
                                       dBarray + i, lddb + i, item_ct1);
                               });
        }
    }
}