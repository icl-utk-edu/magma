/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define COMPLEX



void magma_zlarfg_gpu_kernel( int n, magmaDoubleComplex* dx0, magmaDoubleComplex* dx,
                              magmaDoubleComplex *dtau, double *dxnorm, magmaDoubleComplex* dAkk,
                              sycl::nd_item<3> item_ct1,
                              magmaDoubleComplex *scale)
{
    const int i = item_ct1.get_local_id(2);
    const int j = i + BLOCK_SIZE * item_ct1.get_group(2);

    double xnorm;

    magmaDoubleComplex dxi;

#ifdef REAL
    if ( n <= 1 )
#else
    if ( n <= 0 )
#endif
    {
        /*
        DPCT1064:1193: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        *dtau = MAGMA_Z_ZERO;
        *dAkk = *dx0;
        return;
    }

    if ( j < n-1)
        dxi = dx[j];

    xnorm = *dxnorm;
    magmaDoubleComplex alpha = *dx0;

#ifdef REAL
    if ( xnorm != 0 ) {
        if (i == 0) {  
            double beta  = sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = (beta - alpha) / beta;
            *dAkk  = beta;

            *scale = MAGMA_D_ONE / (alpha - beta);
        }
#else
    double alphar = MAGMA_Z_REAL(alpha);
    double alphai = MAGMA_Z_IMAG(alpha);
    if ( xnorm != 0 || alphai != 0) {
        if (i == 0) {
            double beta =
                sycl::sqrt(alphar * alphar + alphai * alphai + xnorm * xnorm);
            beta = -sycl::copysign(beta, alphar);

            // todo: deal with badly scaled vectors (see lapack's larfg)
            /*
            DPCT1064:1195: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            *dtau = MAGMA_Z_MAKE((beta - alphar) / beta, -alphai / beta);
            /*
            DPCT1064:1196: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            *dAkk = MAGMA_Z_MAKE(beta, 0.);

            /*
            DPCT1064:1197: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            alpha = MAGMA_Z_MAKE(MAGMA_Z_REAL(alpha)- beta, MAGMA_Z_IMAG(alpha));
            /*
            DPCT1064:1198: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            *scale = MAGMA_Z_DIV(MAGMA_Z_ONE, alpha);
        }
#endif

        // scale x
        /*
        DPCT1065:1194: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( xnorm != 0 && j < n-1)
            dx[j] = MAGMA_Z_MUL(dxi, *scale);
    }
    else {
        /*
        DPCT1064:1199: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        *dtau = MAGMA_Z_ZERO;
        *dAkk = *dx0; 
    }
}


/*
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with |beta| = norm( [dx0, dx] ) = dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.  
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hence beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_zlarfg_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dAkk,
    magma_queue_t queue )
{
    sycl::range<3> blocks(1, 1, magma_ceildiv(n, BLOCK_SIZE));
    sycl::range<3> threads(1, 1, BLOCK_SIZE);

    /* recomputing the norm */
    //magmablas_dznrm2_cols(n, 1, dx0, n, dxnorm);
    magmablas_dznrm2_cols(n-1, 1, dx0+1, n, dxnorm, queue);

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            scale_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zlarfg_gpu_kernel(
                                 n, dx0, dx, dtau, dxnorm, dAkk, item_ct1,
                                 scale_acc_ct1.get_pointer());
                         });
    });
}
