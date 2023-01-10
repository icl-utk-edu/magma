/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Ichitaro Yamazaki
       
       @precisions normal z -> s d c

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define BLK_X 64
#define BLK_Y 32

/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to zlaset, zlacpy, zlag2c, clag2z, zgeadd.
*/
static 
void zlacpy_sym_in_full_device(
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
void zlacpy_sym_in_lower_device(
    int m, int n, magma_int_t *rows, magma_int_t *perm,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2); // row
    int iby = item_ct1.get_group(1) * BLK_Y;                            // col

    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n);
    for (int jj=0; jj < n; jj++) {
        perm[rows[2*jj+1]] = rows[2*jj];
    }
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m ) {
        if ( full ) {
            // full block-column, off-diagonal block
            //#pragma unroll
            for( int jj=0; jj < BLK_Y; ++jj ) 
            {
                int j = rows[2*(iby+jj)];
                if (perm[ind] <= j)
                    dB[ind + (iby+jj)*lddb] = MAGMA_Z_CONJ( dA[j + perm[ind]*ldda] );
                else
                    dB[ind + (iby+jj)*lddb] = dA[perm[ind] + j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int jj=0; jj < BLK_Y && iby+jj < n; ++jj ) 
            {
                int j = rows[2*(iby+jj)];
                if (perm[ind] <= j)
                    dB[ind + (iby+jj)*lddb] = MAGMA_Z_CONJ( dA[j + perm[ind]*ldda] );
                else
                    dB[ind + (iby+jj)*lddb] = dA[perm[ind] + j*ldda];
            }
        }
    }
}


/*
    Similar to zlacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to zlaset, zlacpy, zlat2c, clat2z.
*/
static 
void zlacpy_sym_in_upper_device(
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

void zlacpy_sym_in_full_kernel(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    zlacpy_sym_in_full_device(m, n, dA, ldda, dB, lddb, item_ct1);
}


void zlacpy_sym_in_lower_kernel(
    int m, int n, magma_int_t *rows, magma_int_t *perm,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    zlacpy_sym_in_lower_device(m, n, rows, perm, dA, ldda, dB, lddb, item_ct1);
}


void zlacpy_sym_in_upper_kernel(
    int m, int n,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb , sycl::nd_item<3> item_ct1)
{
    zlacpy_sym_in_upper_device(m, n, dA, ldda, dB, lddb, item_ct1);
}


/***************************************************************************//**
    Purpose
    -------
    ZLACPY_SYM_IN copies all or part of a two-dimensional matrix dA to another
    matrix dB.
    
    This is the same as ZLACPY, but adds queue argument.
    
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
            The number of rows that are swapped.  N >= 0.
 
    @param[in]
    rows     INTEGER array, on GPU, dimension (2*n)
             On entry, it stores the new pivots such that rows[i]-th and rows[n+i]-th
             rows are swapped.

    @param[in,out]
    perm     INTEGER array, on GPU, dimension (m)
             On entry, it stores the identity permutation array.
             On exit, it is updated with the new pivots given by rows such that
             i-th row will be the original perm[i]-th row after the pivots are applied. 

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
            On exit, dB = stores the columns after the pivots are applied.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacpy
*******************************************************************************/
extern "C" void
magmablas_zlacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
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
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }

    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, magma_ceildiv(n, BLK_Y), magma_ceildiv(m, BLK_X));

    if ( uplo == MagmaLower ) {
        /*
        DPCT1049:1091: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlacpy_sym_in_lower_kernel(m, n, rows, perm, dA,
                                                          ldda, dB, lddb,
                                                          item_ct1);
                           });
    }
    else if ( uplo == MagmaUpper ) {
        /*
        DPCT1049:1092: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlacpy_sym_in_upper_kernel(m, n, dA, ldda, dB,
                                                          lddb, item_ct1);
                           });
    }
    else {
        /*
        DPCT1049:1093: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlacpy_sym_in_full_kernel(m, n, dA, ldda, dB,
                                                         lddb, item_ct1);
                           });
    }
}