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
#include "commonblas_z.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define COMPLEX


/******************************************************************************/

void magma_zlarfgx_gpu_kernel( int n, magmaDoubleComplex* dx0, magmaDoubleComplex* dx,
                               magmaDoubleComplex *dtau, double *dxnorm,
                               magmaDoubleComplex *dA, int it,
                               sycl::nd_item<3> item_ct1,
                               magmaDoubleComplex *scale, double *xnorm)
{
    const int i = item_ct1.get_local_id(2);
    const int j = i + BLOCK_SIZE * item_ct1.get_group(2);

    magmaDoubleComplex dxi;

    if ( j < n-1 )
        dxi = dx[j];
  
    if ( i == 0 ) {
        *xnorm = *dxnorm;
#ifdef REAL
        double alpha = *dx0;
        double alphai = MAGMA_Z_ZERO;
        if ( (xnorm == 0 && alphai == MAGMA_Z_ZERO ) || n == 1 )
        #else
        magmaDoubleComplex alpha = *dx0;
        double alphar =  MAGMA_Z_REAL(alpha), alphai = MAGMA_Z_IMAG(alpha);
        /*
        DPCT1064:1202: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if ((*xnorm == 0 && alphai == MAGMA_Z_ZERO) || n == 0)
        #endif
        {
            /*
            DPCT1064:1203: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            *dtau = MAGMA_Z_ZERO;
            *dA   = *dx0;
        }
        else {
            #ifdef REAL
                // no need to compute the norm as it is passed as input
                double beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
                beta  = -copysign( beta, alpha );
     
                // todo: deal with badly scaled vectors (see lapack's larfg)
                if (j == 0) {
                    *dtau = (beta - alpha) / beta;
                    //*dx0  = 1.; //cannot be done here because raise condition all threadblock need to read it for alpha
                    *dA   = beta;
                }
    
                scale = 1. / (alpha - beta);
            #else
                // no need to compute the norm as it is passed as input
                double beta = (*xnorm); // sqrt( alphar*alphar + alphai*alphai +
                                        // xnorm*xnorm );
                beta = -sycl::copysign(beta, alphar);

                // todo: deal with badly scaled vectors (see lapack's larfg)
                if (j == 0) {
                    *dtau = MAGMA_Z_MAKE((beta - alphar)/beta, -alphai/beta);
                    //*dx0  = MAGMA_Z_MAKE(  1., 0.); //cannot be done here because raise condition all threadblock need to read it for alpha
                    *dA   = MAGMA_Z_MAKE(beta, 0.);
                }

                alpha = MAGMA_Z_MAKE(MAGMA_Z_REAL(alpha) - beta, MAGMA_Z_IMAG(alpha));
                /*
                DPCT1064:1204: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                *scale = MAGMA_Z_DIV(MAGMA_Z_ONE, alpha);
            #endif
        }
    }

    // scale x
    /*
    DPCT1065:1201: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (*xnorm != 0 && j < n - 1)
        dx[j] = MAGMA_Z_MUL(dxi, *scale);

    if (j < it) {
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_Z_MAKE(0., 0.);
    }
}


/***************************************************************************//**
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with |beta| = norm( [dx0, dx] ) = dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*******************************************************************************/
extern "C" void
magma_zlarfgx_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter, 
    magma_queue_t queue )
{
    sycl::range<3> blocks(1, 1, magma_ceildiv(n, BLOCK_SIZE));
    sycl::range<3> threads(1, 1, BLOCK_SIZE);

    /*
    DPCT1049:1205: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            scale_acc_ct1(cgh);
        sycl::accessor<double, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            xnorm_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zlarfgx_gpu_kernel(
                                 n, dx0, dx, dtau, dxnorm, dA, iter, item_ct1,
                                 scale_acc_ct1.get_pointer(),
                                 xnorm_acc_ct1.get_pointer());
                         });
    });
}


/***************************************************************************//**
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with |beta| = norm( [dx0, dx] ) = dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*******************************************************************************/
extern "C" void
magma_zlarfgtx_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr T,  magma_int_t ldt,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue )
{
    /*  Generate the elementary reflector H(iter)  */
    magma_zlarfgx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter, queue);
    
    if (iter == 0) {
        /*
        DPCT1064:1206: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex tt = MAGMA_Z_ONE;
        magmablas_zlacpy( MagmaFull, 1, 1, dtau, 1, T+iter+iter*ldt, 1, queue );
        magma_zsetmatrix( 1, 1, &tt, 1, dx0, 1, queue );
    }
    else {
        /* Compute the iter-th column of T */
        /*
        DPCT1049:1207: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sum_acc_ct1(sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, iter) *
                                          sycl::range<3>(1, 1, BLOCK_SIZE),
                                      sycl::range<3>(1, 1, BLOCK_SIZE)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_zgemv_kernel3(n, V, ldv, dx0, dwork, dtau,
                                            item_ct1,
                                            sum_acc_ct1.get_pointer());
                    });
            });

        /*
        DPCT1049:1208: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sum_acc_ct1(sycl::range<1>(128), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, iter) *
                                          sycl::range<3>(1, 1, iter),
                                      sycl::range<3>(1, 1, iter)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_ztrmv_kernel2(T, ldt, dwork, T + iter * ldt, dtau,
                                            item_ct1,
                                            sum_acc_ct1.get_pointer());
                    });
            });
    }
}
