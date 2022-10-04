/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 64


/******************************************************************************/
// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
void
zcaxpycp_kernel(
    int m,
    magmaFloatComplex *r,
    magmaDoubleComplex *x,
    const magmaDoubleComplex *b,
    magmaDoubleComplex *w ,
    sycl::nd_item<3> item_ct1)
{
    const int i = item_ct1.get_local_id(2) + item_ct1.get_group(2) * NB;
    if ( i < m ) {
        /*
        DPCT1064:246: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        x[i] = MAGMA_Z_ADD(x[i], x()(MAGMA_C_REAL(r[i]), MAGMA_C_IMAG(r[i])));
        w[i] = b[i];
    }
}


/***************************************************************************//**
    adds   x += r (including conversion to double)  --and--
    copies w = b
*******************************************************************************/
extern "C" void
magmablas_zcaxpycp(
    magma_int_t m,
    magmaFloatComplex_ptr r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b,
    magmaDoubleComplex_ptr w,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(m, NB));
    /*
    DPCT1049:247: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zcaxpycp_kernel(m, r, x, b, w, item_ct1);
                       });
}
