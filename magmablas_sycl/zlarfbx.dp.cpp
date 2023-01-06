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


/******************************************************************************/
extern "C" SYCL_EXTERNAL void
magma_zgemv_kernel1(int m, const magmaDoubleComplex *__restrict__ V, int ldv,
                    const magmaDoubleComplex *__restrict__ c,
                    magmaDoubleComplex *dwork, sycl::nd_item<3> item_ct1,
                    magmaDoubleComplex *sum)
{
    const int i = item_ct1.get_local_id(2);
    const magmaDoubleComplex *dV = V + (item_ct1.get_group(2)) * ldv;

    magmaDoubleComplex lsum;

    /*  lsum := v**H * C  */
    lsum = MAGMA_Z_ZERO;
    for (int j = i; j < m; j += BLOCK_SIZE)
       lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( dV[j] ), c[j] );
    
    sum[i] = lsum;
    magma_sum_reduce<BLOCK_SIZE>(i, sum, item_ct1);

    /*
    DPCT1065:1179: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i == 0)
       dwork[item_ct1.get_group(2)] = sum[0];
}

/******************************************************************************/
/*
    Call 
        magma_zgemv_kernel3<<< n, BLOCK_SIZE, 0, queue->sycl_stream() >>>(m, V, ldv, c, dwork, tau)
    to compute
        ZGEMV( "Conjugate transpose", m, n, -tau[0], V, ldv, c, 1, zero, dwork, 1)
        and to set c[0] to 1.
    i.e., 
        work = -tau[0] V**H c
*/
extern "C" SYCL_EXTERNAL void
magma_zgemv_kernel3(int m, const magmaDoubleComplex *__restrict__ V, int ldv,
                    magmaDoubleComplex *c, magmaDoubleComplex *dwork,
                    magmaDoubleComplex *tau, sycl::nd_item<3> item_ct1,
                    magmaDoubleComplex *sum)
{
    const int i = item_ct1.get_local_id(2);
    const magmaDoubleComplex *dV = V + (item_ct1.get_group(2)) * ldv;

    magmaDoubleComplex lsum;

    if (i == 0)
       c[0] = MAGMA_Z_ONE;           

    /*  lsum := v**H * C  */
    lsum = MAGMA_Z_ZERO;
    for (int j = i; j < m; j += BLOCK_SIZE)
       lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( dV[j] ), c[j] );

    sum[i] = lsum;
    magma_sum_reduce<BLOCK_SIZE>(i, sum, item_ct1);

    /*
    DPCT1065:1180: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i == 0)
       dwork[item_ct1.get_group(2)] = -tau[0] * sum[0];
}


/******************************************************************************/
extern "C" SYCL_EXTERNAL void
magma_zgemv_kernel2(int m, int n, const magmaDoubleComplex *__restrict__ V,
                    int ldv, const magmaDoubleComplex *__restrict__ x,
                    magmaDoubleComplex *c, sycl::nd_item<3> item_ct1)
{
    const int i = item_ct1.get_local_id(2);
    const int j = i + BLOCK_SIZE * item_ct1.get_group(2);
    magmaDoubleComplex lsum;

    V += j;

    lsum = MAGMA_Z_ZERO;
    if (j < m) {
        for (int k=0; k < n; k++)
            lsum += MAGMA_Z_MUL( V[k*ldv], x[k]);
        
        c[j] -= lsum;
    }
}


/******************************************************************************/
/*
    Apply a complex block reflector H to a complex vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V**H
    where T is the complex k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a complex block of
    k elementary reflectors. 
*/
extern "C" void
magma_zlarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr dT, magma_int_t ldt,
    magmaDoubleComplex_ptr c,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue )
{
    /* dwork = V**H c     */
    /*
    DPCT1049:1182: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, k) *
                                               sycl::range<3>(1, 1, BLOCK_SIZE),
                                           sycl::range<3>(1, 1, BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zgemv_kernel1(m, V, ldv, c, dwork, item_ct1,
                                                 sum_acc_ct1.get_pointer());
                         });
    });

    /* dwork = T**H dwork */
    /*
    DPCT1049:1183: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(sycl::range<1>(128), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, k) * sycl::range<3>(1, 1, k),
                              sycl::range<3>(1, 1, k)),
            [=](sycl::nd_item<3> item_ct1) {
                magma_ztrmv_tkernel(dT, ldt, dwork, dwork + k, item_ct1,
                                    sum_acc_ct1.get_pointer());
            });
    });

    /* c = c - V dwork    */
    sycl::range<3> blocks3(1, 1, magma_ceildiv(m, BLOCK_SIZE));
    sycl::range<3> threads3(1, 1, BLOCK_SIZE);
    /*
    DPCT1049:1181: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(blocks3 * threads3, threads3),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zgemv_kernel2(m, k, V, ldv, dwork + k, c,
                                               item_ct1);
                       });
}
