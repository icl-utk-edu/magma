/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Stan Tomov
       @author Mark Gates
       
       @precisions normal z -> s d c
*/
#include "magma_internal.h"

#ifdef HAVE_clBLAS
    #define dA( dev, i_, j_ )  dA[dev], ((i_) + (j_)*ldda)
#else
    #define dA( dev, i_, j_ ) (dA[dev] + (i_) + (j_)*ldda)
#endif

#define hA( i_, j_ ) (hA + (i_) + (j_)*lda)


/***************************************************************************//**
    Copy matrix hA on CPU host to dA, which is distributed
    1D column block cyclic over multiple GPUs.

    @param[in]  ngpu    Number of GPUs over which dAT is distributed.
    @param[in]  m       Number of rows    of matrix hA. m >= 0.
    @param[in]  n       Number of columns of matrix hA. n >= 0.
    @param[in]  nb      Block size. nb > 0.
    @param[in]  hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[out] dA      Array of ngpu pointers, one per GPU, that store the
                        disributed m-by-n matrix A on the GPUs, each of dimension
                        (ldda,nlocal), where nlocal is the columns assigned to each GPU.
    @param[in]  ldda    Leading dimension of each matrix dAT on each GPU. ldda >= m.
    @param[in]  queues  Array of dimension (ngpu), with one queue per GPU.

    @ingroup magma_setmatrix_bcyclic
*******************************************************************************/
extern "C" void
magma_zsetmatrix_1D_col_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaDoubleComplex *hA, magma_int_t lda,
    magmaDoubleComplex_ptr   *dA, magma_int_t ldda,
    magma_queue_t queues[] )
{
    magma_int_t info = 0;
    if ( ngpu < 1 )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( nb < 1 )
        info = -4;
    else if ( lda < m )
        info = -6;
    else if ( ldda < m )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t j, dev, jb;
    
    magma_device_t cdevice;
    magma_getdevice( &cdevice );
    
    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % ngpu;
        magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_zsetmatrix_async( m, jb,
                                hA(0,j), lda,
                                dA( dev, 0, j/(nb*ngpu)*nb ), ldda,
                                queues[dev] );
    }
    
    for( dev=0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
    }
    
    magma_setdevice( cdevice );
}


/***************************************************************************//**
    Copy matrix dA, which is distributed 1D column block cyclic over multiple
    GPUs, to hA on CPU host.

    @param[in]  ngpu    Number of GPUs over which dAT is distributed.
    @param[in]  m       Number of rows    of matrix hA. m >= 0.
    @param[in]  n       Number of columns of matrix hA. n >= 0.
    @param[in]  nb      Block size. nb > 0.
    @param[in]  dA      Array of ngpu pointers, one per GPU, that store the
                        disributed m-by-n matrix A on the GPUs, each of dimension
                        (ldda,nlocal), where nlocal is the columns assigned to each GPU.
    @param[in]  ldda    Leading dimension of each matrix dAT on each GPU. ldda >= m.
    @param[out] hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[in]  queues  Array of dimension (ngpu), with one queue per GPU.

    @ingroup magma_getmatrix_bcyclic
*******************************************************************************/
extern "C" void
magma_zgetmatrix_1D_col_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_const_ptr const *dA, magma_int_t ldda,
    magmaDoubleComplex                 *hA, magma_int_t lda,
    magma_queue_t queues[] )
{
    magma_int_t info = 0;
    if ( ngpu < 1 )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( nb < 1 )
        info = -4;
    else if ( ldda < m )
        info = -6;
    else if ( lda < m )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t j, dev, jb;
    
    magma_device_t cdevice;
    magma_getdevice( &cdevice );
    
    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % ngpu;
        magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_zgetmatrix_async( m, jb,
                                dA( dev, 0, j/(nb*ngpu)*nb ), ldda,
                                hA(0,j), lda,
                                queues[dev] );
    }
    
    for( dev=0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
    }
    
    magma_setdevice( cdevice );
}


/***************************************************************************//**
    Copy matrix hA on CPU host to dA, which is distributed 1D row block cyclic
    over multiple GPUs.

    @param[in]  ngpu    Number of GPUs over which dAT is distributed.
    @param[in]  m       Number of rows    of matrix hA. m >= 0.
    @param[in]  n       Number of columns of matrix hA. n >= 0.
    @param[in]  nb      Block size. nb > 0.
    @param[in]  hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[out] dA      Array of ngpu pointers, one per GPU, that store the
                        disributed m-by-n matrix A on the GPUs, each of dimension (ldda,n).
    @param[in]  ldda    Leading dimension of each matrix dAT on each GPU.
                        ldda >= (1 + m/(nb*ngpu))*nb
    @param[in]  queues  Array of dimension (ngpu), with one queue per GPU.

    @ingroup magma_setmatrix_bcyclic
*******************************************************************************/
extern "C" void
magma_zsetmatrix_1D_row_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaDoubleComplex    *hA, magma_int_t lda,
    magmaDoubleComplex_ptr      *dA, magma_int_t ldda,
    magma_queue_t queues[] )
{
    magma_int_t info = 0;
    if ( ngpu < 1 )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( nb < 1 )
        info = -4;
    else if ( lda < m )
        info = -6;
    else if ( ldda < (1+m/(nb*ngpu))*nb )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t i, dev, jb;
    
    magma_device_t cdevice;
    magma_getdevice( &cdevice );
    
    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % ngpu;
        magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_zsetmatrix_async( jb, n,
                                hA(i,0), lda,
                                dA( dev, i/(nb*ngpu)*nb, 0 ), ldda,
                                queues[dev] );
    }
    
    for( dev=0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
    }
    
    magma_setdevice( cdevice );
}


/***************************************************************************//**
    Copy matrix dA, which is distributed 1D row block cyclic over multiple
    GPUs, to hA on CPU host.

    @param[in]  ngpu    Number of GPUs over which dAT is distributed. ngpu > 0.
    @param[in]  m       Number of rows    of matrix hA. m >= 0.
    @param[in]  n       Number of columns of matrix hA. n >= 0.
    @param[in]  nb      Block size. nb > 0.
    @param[in]  dA      Array of ngpu pointers, one per GPU, that store the
                        disributed m-by-n matrix A on the GPUs, each of dimension (ldda,n).
    @param[in]  ldda    Leading dimension of each matrix dAT on each GPU.
                        ldda >= (1 + m/(nb*ngpu))*nb
    @param[out] hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[in]  queues  Array of dimension (ngpu), with one queue per GPU.

    @ingroup magma_getmatrix_bcyclic
*******************************************************************************/
extern "C" void
magma_zgetmatrix_1D_row_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_const_ptr const *dA, magma_int_t ldda,
    magmaDoubleComplex                 *hA, magma_int_t lda,
    magma_queue_t queues[] )
{
    magma_int_t info = 0;
    if ( ngpu < 1 )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( nb < 1 )
        info = -4;
    else if ( ldda < (1+m/(nb*ngpu))*nb )
        info = -6;
    else if ( lda < m )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t i, dev, jb;
    
    magma_device_t cdevice;
    magma_getdevice( &cdevice );
    
    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % ngpu;
        magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_zgetmatrix_async( jb, n,
                                dA( dev, i/(nb*ngpu)*nb, 0 ), ldda,
                                hA(i,0), lda,
                                queues[dev] );
    }
    
    for( dev=0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
    }
    
    magma_setdevice( cdevice );
}
