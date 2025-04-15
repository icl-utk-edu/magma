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
static 
void zlaset_full_device(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
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
static 
void zlaset_lower_device(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
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
static 
void zlaset_upper_device(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    int iby = item_ct1.get_group(1) * BLK_Y;
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

void zlaset_full_kernel(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, int ldda , sycl::nd_item<3> item_ct1)
{
    zlaset_full_device(m, n, offdiag, diag, dA, ldda, item_ct1);
}


void zlaset_lower_kernel(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, int ldda , sycl::nd_item<3> item_ct1)
{
    zlaset_lower_device(m, n, offdiag, diag, dA, ldda, item_ct1);
}


void zlaset_upper_kernel(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, int ldda , sycl::nd_item<3> item_ct1)
{
    zlaset_upper_device(m, n, offdiag, diag, dA, ldda, item_ct1);
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions for the batched routine.
*/

void zlaset_full_kernel_batched(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, int Ai, int Aj, int ldda ,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    magmaDoubleComplex *dA = dAarray[batchid] + Aj * ldda + Ai;
    zlaset_full_device(m, n, offdiag, diag, dA, ldda, item_ct1);
}


void zlaset_lower_kernel_batched(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, int Ai, int Aj, int ldda ,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    magmaDoubleComplex *dA = dAarray[batchid] + Aj * ldda + Ai;
    zlaset_lower_device(m, n, offdiag, diag, dA, ldda, item_ct1);
}


void zlaset_upper_kernel_batched(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, int Ai, int Aj, int ldda ,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    magmaDoubleComplex *dA = dAarray[batchid] + Aj * ldda + Ai;
    zlaset_upper_device(m, n, offdiag, diag, dA, ldda, item_ct1);
}
/******************************************************************************/
/*
    kernel wrappers to call the device functions for the vbatched routine.
*/

void zlaset_full_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, magma_int_t* ldda , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= (my_m + BLK_X - 1) / BLK_X) return;
    if (item_ct1.get_group(1) >= (my_n + BLK_Y - 1) / BLK_Y) return;
    zlaset_full_device(my_m, my_n, offdiag, diag, dAarray[batchid],
                       (int)ldda[batchid], item_ct1);
}


void zlaset_lower_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, magma_int_t* ldda , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= (my_m + BLK_X - 1) / BLK_X) return;
    if (item_ct1.get_group(1) >= (my_n + BLK_Y - 1) / BLK_Y) return;
    zlaset_lower_device(my_m, my_n, offdiag, diag, dAarray[batchid],
                        (int)ldda[batchid], item_ct1);
}


void zlaset_upper_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex **dAarray, magma_int_t* ldda , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= (my_m + BLK_X - 1) / BLK_X) return;
    if (item_ct1.get_group(1) >= (my_n + BLK_Y - 1) / BLK_Y) return;
    zlaset_upper_device(my_m, my_n, offdiag, diag, dAarray[batchid],
                        (int)ldda[batchid], item_ct1);
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
extern "C" void magmablas_zlaset(magma_uplo_t uplo, magma_int_t m,
                                 magma_int_t n, magmaDoubleComplex offdiag,
                                 magmaDoubleComplex diag,
                                 magmaDoubleComplex_ptr dA, magma_int_t ldda,
                                 magma_queue_t queue) try {
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
    sycl::range<3> super_grid(1, magma_ceildiv(n, super_NB),
                              magma_ceildiv(m, super_NB));

    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, 1);

    magma_int_t mm, nn;
    if (uplo == MagmaLower) {
        for (unsigned int i = 0; i < super_grid[2]; ++i) {
            mm = (i == super_grid[2] - 1 ? m % super_NB : super_NB);
            grid[2] = magma_ceildiv(mm, BLK_X);
            for (unsigned int j = 0; j < super_grid[1] && j <= i;
                 ++j) { // from left to diagonal
                nn = (j == super_grid[1] - 1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv(nn, BLK_Y);
                if ( i == j ) {  // diagonal super block
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlaset_lower_kernel(
                                    mm, nn, offdiag, diag,
                                    dA(i * super_NB, j * super_NB), ldda,
                                    item_ct1);
                            });
                }
                else {           // off diagonal super block
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlaset_full_kernel(
                                    mm, nn, offdiag, offdiag,
                                    dA(i * super_NB, j * super_NB), ldda,
                                    item_ct1);
                            });
                }
            }
        }
    }
    else if (uplo == MagmaUpper) {
        for (unsigned int i = 0; i < super_grid[2]; ++i) {
            mm = (i == super_grid[2] - 1 ? m % super_NB : super_NB);
            grid[2] = magma_ceildiv(mm, BLK_X);
            for (unsigned int j = i; j < super_grid[1];
                 ++j) { // from diagonal to right
                nn = (j == super_grid[1] - 1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv(nn, BLK_Y);
                if ( i == j ) {  // diagonal super block
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlaset_upper_kernel(
                                    mm, nn, offdiag, diag,
                                    dA(i * super_NB, j * super_NB), ldda,
                                    item_ct1);
                            });
                }
                else {           // off diagonal super block
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zlaset_full_kernel(
                                    mm, nn, offdiag, offdiag,
                                    dA(i * super_NB, j * super_NB), ldda,
                                    item_ct1);
                            });
                }
            }
        }
    }
    else {
        // if continuous in memory & set to zero, cudaMemset is faster.
        // TODO: use cudaMemset2D ?
        if (m == ldda &&
            /*
            DPCT1064:1249: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            MAGMA_Z_EQUAL(offdiag, MAGMA_Z_ZERO) &&
            /*
            DPCT1064:1250: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            MAGMA_Z_EQUAL(diag, MAGMA_Z_ZERO))
        {
            size_t size = m*n;
            /*
            DPCT1003:1251: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            int err = (queue->sycl_stream()->memset(
                           dA, 0, size * sizeof(magmaDoubleComplex)),
                       0);
            assert(err == 0);
            MAGMA_UNUSED( err );
        }
        else {
            for (unsigned int i = 0; i < super_grid[2]; ++i) {
                mm = (i == super_grid[2] - 1 ? m % super_NB : super_NB);
                grid[2] = magma_ceildiv(mm, BLK_X);
                for (unsigned int j = 0; j < super_grid[1]; ++j) { // full row
                    nn = (j == super_grid[1] - 1 ? n % super_NB : super_NB);
                    grid[1] = magma_ceildiv(nn, BLK_Y);
                    if ( i == j ) {  // diagonal super block
                        ((sycl::queue *)(queue->sycl_stream()))
                            ->parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    zlaset_full_kernel(
                                        mm, nn, offdiag, diag,
                                        dA(i * super_NB, j * super_NB), ldda,
                                        item_ct1);
                                });
                    }
                    else {           // off diagonal super block
                        ((sycl::queue *)(queue->sycl_stream()))
                            ->parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    zlaset_full_kernel(
                                        mm, nn, offdiag, offdiag,
                                        dA(i * super_NB, j * super_NB), ldda,
                                        item_ct1);
                                });
                    }
                }
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/******************************************************************************/
extern "C"
void magmablas_zlaset_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, magma_ceildiv(n, BLK_Y),
                            magma_ceildiv(m, BLK_X));

        if (uplo == MagmaLower) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlaset_lower_kernel_batched(
                                       m, n, offdiag, diag, dAarray + i, Ai, Aj,
                                       ldda, item_ct1);
                               });
        }
        else if (uplo == MagmaUpper) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlaset_upper_kernel_batched(
                                       m, n, offdiag, diag, dAarray + i, Ai, Aj,
                                       ldda, item_ct1);
                               });
        }
        else {
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlaset_full_kernel_batched(
                                       m, n, offdiag, diag, dAarray + i, Ai, Aj,
                                       ldda, item_ct1);
                               });
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

    magmablas_zlaset_internal_batched(
            uplo, m, n, offdiag, diag,
            dAarray, 0, 0, ldda,
            batchCount, queue);
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

    sycl::range<3> threads(1, 1, BLK_X);
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, magma_ceildiv(max_n, BLK_Y),
                            magma_ceildiv(max_m, BLK_X));

        if (uplo == MagmaLower) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlaset_lower_kernel_vbatched(
                                       m + i, n + i, offdiag, diag, dAarray + i,
                                       ldda + i, item_ct1);
                               });
        }
        else if (uplo == MagmaUpper) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlaset_upper_kernel_vbatched(
                                       m + i, n + i, offdiag, diag, dAarray + i,
                                       ldda + i, item_ct1);
                               });
        }
        else {
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zlaset_full_kernel_vbatched(
                                       m + i, n + i, offdiag, diag, dAarray + i,
                                       ldda + i, item_ct1);
                               });
        }
    }
}
