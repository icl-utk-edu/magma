/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define NB 64


/* Vector is divided into ceil(n/nb) blocks.
   Each thread swaps one element, x[tid] <---> y[tid].
*/
__global__ void zswap_kernel(
    int n,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex *y, int incy )
{
    magmaDoubleComplex tmp;
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    if ( ind < n ) {
        x += ind*incx;
        y += ind*incy;
        tmp = *x;
        *x  = *y;
        *y  = tmp;
    }
}


/***************************************************************************//**
    Purpose:
    =============
    Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swap
*******************************************************************************/
extern "C" void 
magmablas_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx, 
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( n, NB ) );
    zswap_kernel<<< grid, threads, 0, queue->cuda_stream() >>>( n, dx, incx, dy, incy );
}
