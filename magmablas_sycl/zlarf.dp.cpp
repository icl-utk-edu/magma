/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Azzam Haidar

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


/******************************************************************************/

void magma_zlarf_kernel(
    int m, const magmaDoubleComplex *dv, const magmaDoubleComplex *dtau,
    magmaDoubleComplex *dc, int lddc , sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *sum)
{
    if ( !MAGMA_Z_EQUAL(*dtau, MAGMA_Z_ZERO) ) {
        const int tx = item_ct1.get_local_id(2);
        dc = dc + item_ct1.get_group(2) * lddc;

        magmaDoubleComplex tmp;

        /* perform  w := v**H * C  */
        if (tx == 0)
            tmp = dc[0]; //since V[0] should be one
        else
            /*
            DPCT1064:1185: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            tmp = MAGMA_Z_ZERO;
        for( int j = tx+1; j < m; j += BLOCK_SIZE ) {
            tmp += MAGMA_Z_MUL( MAGMA_Z_CONJ( dv[j] ), dc[j] );
        }
        sum[tx] = tmp;
        magma_sum_reduce<BLOCK_SIZE>(tx, sum, item_ct1);

        /*  C := C - v * w  */
        /*
        DPCT1065:1184: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = - MAGMA_Z_CONJ(*dtau) * sum[0];
        for( int j = m-tx-1; j > 0; j -= BLOCK_SIZE )
             dc[j] += tmp * dv[j];

        if (tx == 0) dc[0] += tmp;
    }
}


/******************************************************************************/

void magma_zlarf_smkernel(
    int m, int n, magmaDoubleComplex *dv, magmaDoubleComplex *dtau,
    magmaDoubleComplex *dc, int lddc , sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sum)
{
    if ( ! MAGMA_Z_EQUAL(*dtau, MAGMA_Z_ZERO) ) {
        const int i = item_ct1.get_local_id(2), col = item_ct1.get_local_id(1);

        for( int k = col; k < n; k += BLOCK_SIZEy ) {
            dc = dc + k * lddc;

            magmaDoubleComplex lsum;
    
            /*  w := v**H * C  */
            /*
            DPCT1064:1187: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            lsum = MAGMA_Z_ZERO;
            for( int j = i; j < m; j += BLOCK_SIZEx ) {
                if (j == 0)
                   lsum += MAGMA_Z_MUL( MAGMA_Z_ONE, dc[j] );
                else
                   lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( dv[j] ), dc[j] );
            }
            sum[i][col] = lsum;
            magma_sum_reduce_2d<BLOCK_SIZEx, BLOCK_SIZEy + 1>(i, col, sum,
                                                              item_ct1);

            /*  C := C - v * w  */
            /*
            DPCT1065:1186: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            magmaDoubleComplex z__1 = - MAGMA_Z_CONJ(*dtau) * sum[0][col];
            for( int j = m-i-1; j >= 0; j -= BLOCK_SIZEx ) {
                if (j == 0)
                    dc[j] += z__1;
                else
                    dc[j] += z__1 * dv[j];
            }
        }
    }
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

    This routine uses only one SM (block).
*/
extern "C" void
magma_zlarf_sm(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *dv, magmaDoubleComplex *dtau,
    magmaDoubleComplex *dc, magma_int_t lddc,
    magma_queue_t queue )
{
    sycl::range<3> blocks(1, 1, 1);
    sycl::range<3> threads(1, BLOCK_SIZEy, BLOCK_SIZEx);

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(
                sycl::range<2>(BLOCK_SIZEx, BLOCK_SIZEy + 1),
                cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zlarf_smkernel(m, n, dv, dtau, dc, lddc,
                                                  item_ct1, sum_acc_ct1);
                         });
    });
}


/***************************************************************************//**
    Apply a complex elementary reflector H to a complex M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v**H
    where tau is a complex scalar and v is a complex vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H**H (the conjugate transpose of H), supply conjg(tau) 
    instead tau.
*******************************************************************************/
extern "C" magma_int_t
magma_zlarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaDoubleComplex_const_ptr dv,
    magmaDoubleComplex_const_ptr dtau,
    magmaDoubleComplex_ptr dC,  magma_int_t lddc,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, n);
    sycl::range<3> threads(1, 1, BLOCK_SIZE);
    if ( n > 0 ) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zlarf_kernel(
                                         m, dv, dtau, dC, lddc, item_ct1,
                                         sum_acc_ct1.get_pointer());
                                 });
            });
    }

    // The computation can be done on 1 SM with the following routine.
    // magma_zlarf_sm(m, n, dv, dtau, dc, lddc);

    return MAGMA_SUCCESS;
}
