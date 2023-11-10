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
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


/******************************************************************************/

void magma_zlarfx_kernel( int m, magmaDoubleComplex *v, magmaDoubleComplex *tau,
                         magmaDoubleComplex *c, int ldc, double *xnorm,
                         magmaDoubleComplex *T, int it ,
                         sycl::nd_item<3> item_ct1, magmaDoubleComplex *sum)
{
    if ( !MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO) ) {
        const int tx = item_ct1.get_local_id(2);
        //magmaDoubleComplex *dc = c + (blockIdx.x-it-1) * ldc;
        magmaDoubleComplex *dc = c + (item_ct1.get_group(2)) * ldc;

        magmaDoubleComplex lsum;

        /* NOTE HERE C is the C at position C(i, 0) 
         * if blockIdx.x < it it performs the V(i:n,i)' * V(i:n,1:i-1)' used for computing T
         * if blockIdx.x > it it perform  w := v**H * C  */
        lsum = MAGMA_Z_ZERO;
        for (int j = tx; j < m; j += BLOCK_SIZE) {
            if (j == 0) {
                lsum += MAGMA_Z_MUL( MAGMA_Z_ONE, dc[j] );
                v[j] = MAGMA_Z_ONE;
            }
            else
                lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( v[j] ), dc[j] );
        }
        sum[tx] = lsum;
        magma_sum_reduce<BLOCK_SIZE>(tx, sum, item_ct1);

        /*  C := C - v * w  */
        /*
        DPCT1065:1224: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        magmaDoubleComplex z__1 = - MAGMA_Z_CONJ(*tau) * sum[0];
        if (item_ct1.get_group(2) > it) {
            for (int j = m-tx-1; j >= 0; j -= BLOCK_SIZE)
                 dc[j] += z__1 * v[j];
             /*
             DPCT1065:1225: Consider replacing sycl::nd_item::barrier() with
             sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
             better performance if there is no access to global memory.
             */
             item_ct1.barrier();

            /* Adjust the rest of the column norms */
            /*
            if (tx == 0) {
                double temp = MAGMA_Z_ABS( dc[0] ) / xnorm[blockIdx.x-it-1];
                temp = (temp + 1.) * (1. - temp);
                xnorm[blockIdx.x-it-1] = xnorm[blockIdx.x-it-1] * sqrt(temp); 
            }
            */
        }
        else
        {
            if (item_ct1.get_group(2) == it)
                *(T+it) = *tau;
            else
                *(T + item_ct1.get_group(2)) = MAGMA_Z_CONJ(z__1);
        }
    } else if (item_ct1.get_group(2) <= it) // in case tau is zero put the
                                            // corresponding column of T to zero
    {
        *(T + item_ct1.get_group(2)) = MAGMA_Z_ZERO;
    }
}


/******************************************************************************/
extern "C"

void magma_ztrmv_kernel(const magmaDoubleComplex *T, int ldt, magmaDoubleComplex *t,
                        sycl::nd_item<3> item_ct1, magmaDoubleComplex *tlocal)
{
    const int tx = item_ct1.get_local_id(2);
    T += tx;

    /*
    DPCT1064:1227: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex res = MAGMA_Z_MAKE(0., 0.);

    tlocal[tx] = t[tx];
    /*
    DPCT1065:1226: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for (int j = 0; j < item_ct1.get_local_range(2); j++)
       res +=  T[j*ldt]*tlocal[j];
    
    t[tx] = res;
}


/******************************************************************************/
extern "C" SYCL_EXTERNAL void
magma_ztrmv_kernel2(const magmaDoubleComplex *T, int ldt, magmaDoubleComplex *t,
                    magmaDoubleComplex *y, magmaDoubleComplex *tau,
                    sycl::nd_item<3> item_ct1, magmaDoubleComplex *sum)
{
    const int tx = item_ct1.get_local_id(2);
    T += item_ct1.get_group(2);

    sum[tx] = T[tx*ldt]*t[tx];
    magma_sum_reduce_n(item_ct1.get_local_range(2), tx, sum, item_ct1);

    /*
    DPCT1065:1228: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (tx == 0) {
        y[item_ct1.get_group(2)] = sum[0];
        if (item_ct1.get_group(2) == 0)
            y[item_ct1.get_group_range(2)] = tau[0];
    }
}


/******************************************************************************/
extern "C" SYCL_EXTERNAL void
magma_ztrmv_tkernel(magmaDoubleComplex *T, int ldt, magmaDoubleComplex *t,
                    magmaDoubleComplex *y, sycl::nd_item<3> item_ct1,
                    magmaDoubleComplex *sum)
{
    const int tx = item_ct1.get_local_id(2);
    T += item_ct1.get_group(2) * ldt;

    sum[tx] = MAGMA_Z_CONJ(T[tx])*t[tx];
    magma_sum_reduce_n(item_ct1.get_local_range(2), tx, sum, item_ct1);

    /*
    DPCT1065:1229: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (tx == 0)
        y[item_ct1.get_group(2)] = sum[0];
}


/******************************************************************************/
/*
    Apply a complex elementary reflector H to a complex M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v**H
    where tau is a complex scalar and v is a complex vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H**H (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

    The norms of v(:, 1:n) are given as input in xnorm(1:n). On exit, the norms
    are adjusted to hold the norms of v(2:m,2:n). This is a difference with the 
    LAPACK's zlarf routine. 
*/
extern "C" void
magma_zlarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr tau,
    magmaDoubleComplex_ptr C, magma_int_t ldc,
    magmaDouble_ptr        xnorm, 
    magmaDoubleComplex_ptr dT, magma_int_t iter,
    magmaDoubleComplex_ptr work,
    magma_queue_t queue )
{
    magma_int_t N = n + iter + 1;

    if (iter == 0) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, N) *
                                          sycl::range<3>(1, 1, BLOCK_SIZE),
                                      sycl::range<3>(1, 1, BLOCK_SIZE)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_zlarfx_kernel(m, v, tau, C, ldc, xnorm,
                                            dT + iter * N, iter, item_ct1,
                                            sum_acc_ct1.get_pointer());
                    });
            });
    }
    else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, N) *
                                          sycl::range<3>(1, 1, BLOCK_SIZE),
                                      sycl::range<3>(1, 1, BLOCK_SIZE)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_zlarfx_kernel(m, v, tau, C, ldc, xnorm, work,
                                            iter, item_ct1,
                                            sum_acc_ct1.get_pointer());
                    });
            });
    }

    if (iter > 0) {
        //magma_ztrmv_kernel
        //    <<< 1, iter, 0, queue->sycl_stream() >>>
        //    ( dT, N, dT+iter*N);
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
                        magma_ztrmv_kernel2(dT, N, work, dT + iter * N, tau,
                                            item_ct1,
                                            sum_acc_ct1.get_pointer());
                    });
            });
    }
}
