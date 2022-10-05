/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 64


/* Vector is divided into ceil(n/nb) blocks.
   Each thread swaps one element, x[tid] <---> y[tid].
*/
void zswap_kernel(
    int n,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex *y, int incy , sycl::nd_item<3> item_ct1)
{
    magmaDoubleComplex tmp;
    int ind = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);
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
    sycl::range<3> threads(1, 1, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, NB));
    /*
    DPCT1049:1329: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zswap_kernel(n, dx, incx, dy, incy, item_ct1);
                       });
}
