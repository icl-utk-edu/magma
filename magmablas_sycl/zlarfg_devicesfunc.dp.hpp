/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_templates.h"

#ifndef MAGMABLAS_ZLARFG_DEVICES_Z_H
#define MAGMABLAS_ZLARFG_DEVICES_Z_H

#define COMPLEX

/******************************************************************************/
/*
    lapack zlarfg, compute the norm, scale and generate the householder vector   
    assume swork, sscale, scale are already allocated in shared memory
    BLOCK_SIZE is set outside, the size of swork is BLOCK_SIZE
*/
static void
zlarfg_device(
    magma_int_t n,
    magmaDoubleComplex* dalpha, magmaDoubleComplex* dx, int incx,
    magmaDoubleComplex* dtau,  double* swork, double* sscale, magmaDoubleComplex* scale,
    sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);

    magmaDoubleComplex tmp;
    
    // find max of [dalpha, dx], to use as scaling to avoid unnecesary under- and overflow    

    if ( tx == 0 ) {
        tmp = *dalpha;
        #ifdef COMPLEX
        swork[tx] = max( fabs(real(tmp)), fabs(imag(tmp)) );
        #else
        swork[tx] = fabs(tmp);
        #endif
    }
    else {
        swork[tx] = 0;
    }
    if (tx < BLOCK_SIZE)
    {
        for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
            tmp = dx[j*incx];
            #ifdef COMPLEX
            swork[tx] = max( swork[tx], max( fabs(real(tmp)), fabs(imag(tmp)) ));
            #else
            swork[tx] = max( swork[tx], fabs(tmp) );
            #endif
        }
    }

    magma_max_reduce<BLOCK_SIZE>( tx, swork, item_ct1 );

    if ( tx == 0 )
        *sscale = swork[0];
    /*
    DPCT1065:156: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // sum norm^2 of dx/sscale
    // dx has length n-1
    if (tx < BLOCK_SIZE) swork[tx] = 0;
    if ( *sscale > 0 ) {
        if (tx < BLOCK_SIZE)
        {
            for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
                tmp = dx[j*incx] / *sscale;
                swork[tx] += real(tmp)*real(tmp) + imag(tmp)*imag(tmp);
            }
        }
        magma_sum_reduce<BLOCK_SIZE>( tx, swork, item_ct1 );
    }
    
    if ( tx == 0 ) {
        magmaDoubleComplex alpha = *dalpha;

        if ( swork[0] == 0 && imag(alpha) == 0 ) {
            // H = I
            *dtau = MAGMA_Z_ZERO;
        }
        else {
            // beta = norm( [dalpha, dx] )
            double beta;
            tmp  = alpha / *sscale;
            beta = *sscale * sqrt( real(tmp)*real(tmp) + imag(tmp)*imag(tmp) + swork[0] );
            beta = -copysign( beta, real(alpha) );
            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau   = MAGMA_Z_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );
            *dalpha = MAGMA_Z_MAKE( beta, 0 );
            *scale = 1 / (alpha - beta);
        }
    }
    
    // scale x (if norm was not 0)
    /*
    DPCT1065:157: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if ( swork[0] != 0 ) {
        if (tx < BLOCK_SIZE)
        {
            for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
                dx[j*incx] *= *scale;
            }
        }
    }
}

#endif // MAGMABLAS_ZLARFG_DEVICES_Z_H
