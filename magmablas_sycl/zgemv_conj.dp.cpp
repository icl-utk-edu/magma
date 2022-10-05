/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Stan Tomov

       @precisions normal z -> s d c
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"

#define num_threads 256


void
zgemv_conj_kernel(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex *       __restrict__ y, int incy,
    sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * num_threads + item_ct1.get_local_id(2);

    A += ind;

    if ( ind < m ) {
        magmaDoubleComplex res = MAGMA_Z_ZERO;
        
        #pragma unroll
        for( int i=0; i < n; i ++ ) {
            res += A[0] * MAGMA_Z_CONJ(x[0]);
            A += lda;
            x += incx;
        }
        
        y[ind*incy] = alpha * res + beta * y[ind*incy];
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEMV_CONJ performs the matrix-vector operation
    
        y := alpha*A*conj(x)    + beta*y, 
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A

    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      COMPLEX_16 array of dimension ( LDDA, n ) on the GPU.

    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx      COMPLEX_16 array of dimension n

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    beta    DOUBLE REAL
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      DOUBLE PRECISION array of dimension m

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv
*******************************************************************************/
extern "C" void
magmablas_zgemv_conj(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -5;
    else if ( incx == 0 )
        info = -7;
    else if ( incy == 0 )
        info = -10;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t blocks = magma_ceildiv( m, num_threads );
    sycl::range<3> grid(1, 1, blocks);
    sycl::range<3> threads(1, 1, num_threads);

    /*
    DPCT1049:362: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgemv_conj_kernel(m, n, alpha, dA, ldda, dx, incx,
                                             beta, dy, incy, item_ct1);
                       });
}
