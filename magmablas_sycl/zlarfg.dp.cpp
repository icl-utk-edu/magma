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
#include "magma_templates.h"

#define COMPLEX

// 512 is maximum number of threads for CUDA capability 1.x
#define NB 512


/******************************************************************************/
// kernel for magma_zlarfg.
// Uses one block of NB (currently 512) threads.
// Each thread sums dx[ tx + k*NB ]^2 for k = 0, 1, ...,
// then does parallel sum reduction to get norm-squared.
// 
// Currently setup to use NB threads, no matter how small dx is.
// This was slightly faster (5%) than passing n to magma_sum_reduce.
// To use number of threads = min( NB, max( 1, n-1 )), pass n as
// argument to magma_sum_reduce, rather than as template parameter.
void
zlarfg_kernel(
    int n,
    magmaDoubleComplex* dalpha,
    magmaDoubleComplex* dx, int incx,
    magmaDoubleComplex* dtau , sycl::nd_item<3> item_ct1, double *swork,
    double *sscale, magmaDoubleComplex *sscale2)
{
    const int tx = item_ct1.get_local_id(2);

    // TODO is it faster for each thread to have its own scale (register)?
    // if so, communicate it via swork[0]

    magmaDoubleComplex tmp;
    
    // find max of [dalpha, dx], to use as scaling to avoid unnecesary under- and overflow
    if ( tx == 0 ) {
        tmp = *dalpha;
        #ifdef COMPLEX
        swork[tx] = max(sycl::fabs(MAGMA_Z_REAL(tmp)), sycl::fabs(MAGMA_Z_IMAG(tmp)));
#else
        swork[tx] = fabs(tmp);
        #endif
    }
    else {
        swork[tx] = 0;
    }
    for( int j = tx; j < n-1; j += NB ) {
        tmp = dx[j*incx];
        #ifdef COMPLEX
        swork[tx] = max(swork[tx], max(sycl::fabs(MAGMA_Z_REAL(tmp)),
                                       sycl::fabs(MAGMA_Z_IMAG(tmp))));
#else
        swork[tx] = max( swork[tx], fabs(tmp) );
        #endif
    }
    magma_max_reduce<NB>(tx, swork, item_ct1);
    if ( tx == 0 )
        *sscale = swork[0];
    /*
    DPCT1065:1190: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // sum norm^2 of dx/sscale
    // dx has length n-1
    swork[tx] = 0;
    if (*sscale > 0) {
        for( int j = tx; j < n-1; j += NB ) {
            tmp = dx[j * incx] / *sscale;
            swork[tx] += MAGMA_Z_REAL(tmp)*MAGMA_Z_REAL(tmp) + MAGMA_Z_IMAG(tmp)*MAGMA_Z_IMAG(tmp);
        }
        magma_sum_reduce<NB>(tx, swork, item_ct1);
        //magma_sum_reduce( blockDim.x, tx, swork );
    }
    
    if ( tx == 0 ) {
        magmaDoubleComplex alpha = *dalpha;
        if ( swork[0] == 0 && MAGMA_Z_IMAG(alpha) == 0 ) {
            // H = I
            *dtau = MAGMA_Z_ZERO;
        }
        else {
            // beta = norm( [dalpha, dx] )
            double beta;
            tmp = alpha / *sscale;
            beta = *sscale * sycl::sqrt(MAGMA_Z_REAL(tmp) * MAGMA_Z_REAL(tmp) +
                                        MAGMA_Z_IMAG(tmp) * MAGMA_Z_IMAG(tmp) + swork[0]);
            beta = -sycl::copysign(beta, MAGMA_Z_REAL(alpha));
            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau =
                MAGMA_Z_MAKE((beta - MAGMA_Z_REAL(alpha)) / beta, -MAGMA_Z_IMAG(alpha) / beta);
            *dalpha = MAGMA_Z_MAKE( beta, 0 );
            *sscale2 = 1 / (alpha - beta);
        }
    }
    
    // scale x (if norm was not 0)
    /*
    DPCT1065:1191: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( swork[0] != 0 ) {
        for( int j = tx; j < n-1; j += NB ) {
            dx[j * incx] *= *sscale2;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLARFG generates a complex elementary reflector (Householder matrix)
    H of order n, such that

         H * ( alpha ) = ( beta ),   H**H * H = I.
             (   x   )   (   0  )

    where alpha and beta are scalars, with beta real and |beta| = norm([alpha, x]),
    and x is an (n-1)-element complex vector. H is represented in the form

         H = I - tau * ( 1 ) * ( 1 v**H ),
                       ( v )

    where tau is a complex scalar and v is a complex (n-1)-element vector.
    Note that H is not Hermitian.

    If the elements of x are all zero and dalpha is real, then tau = 0
    and H is taken to be the unit matrix.

    Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the elementary reflector.

    @param[in,out]
    dalpha  COMPLEX_16* on the GPU.
            On entry, pointer to the value alpha, i.e., the first entry of the vector.
            On exit, it is overwritten with the value beta.

    @param[in,out]
    dx      COMPLEX_16 array, dimension (1+(N-2)*abs(INCX)), on the GPU
            On entry, the (n-1)-element vector x.
            On exit, it is overwritten with the vector v.

    @param[in]
    incx    INTEGER
            The increment between elements of X. INCX > 0.

    @param[out]
    dtau    COMPLEX_16* on the GPU.
            Pointer to the value tau.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_larfg
*******************************************************************************/
extern "C"
void magmablas_zlarfg(
    magma_int_t n,
    magmaDoubleComplex_ptr dalpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dtau,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, NB);
    sycl::range<3> blocks(1, 1, 1);
    /*
    DPCT1049:1192: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            swork_acc_ct1(sycl::range<1>(NB), cgh);
        sycl::accessor<double, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sscale_acc_ct1(cgh);
        sycl::accessor<magmaDoubleComplex, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sscale2_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zlarfg_kernel(n, dalpha, dx, incx, dtau, item_ct1,
                                           swork_acc_ct1.get_pointer(),
                                           sscale_acc_ct1.get_pointer(),
                                           sscale2_acc_ct1.get_pointer());
                         });
    });
}
