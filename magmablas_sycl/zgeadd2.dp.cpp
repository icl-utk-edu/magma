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

#define BLK_X 64
#define BLK_Y 32

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to zlaset.
*/

void zgeadd2_full(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex beta,
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
                dB[j*lddb] = alpha*dA[j*ldda] + beta*dB[j*lddb];
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = alpha*dA[j*ldda] + beta*dB[j*lddb];
            }
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEADD2 adds two matrices, dB = alpha*dA + beta*dB.
    
    @see ZGEADD for dB = alpha*dA + dB, lacking beta.
    
    Arguments
    ---------
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    alpha   COMPLEX_16
            The scalar alpha.
            
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The m by n matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    beta    COMPLEX_16
            The scalar beta.
            
    @param[in,out]
    dB      COMPLEX_16 array, dimension (LDDB,N)
            The m by n matrix dB.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_geadd
*******************************************************************************/
extern "C" void
magmablas_zgeadd2(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
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
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;

    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, magma_ceildiv(n, BLK_Y), magma_ceildiv(m, BLK_X));

    /*
    DPCT1049:275: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgeadd2_full(m, n, alpha, dA, ldda, beta, dB, lddb,
                                        item_ct1);
                       });
}
