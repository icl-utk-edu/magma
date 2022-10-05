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
#define BLOCK_SIZE  512
#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

#define COMPLEX


/******************************************************************************/
void
magmablas_dznrm2_kernel(
    int m,
    magmaDoubleComplex *dA, int ldda,
    double *dxnorm , sycl::nd_item<3> item_ct1, double *sum)
{
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex *dx = dA + item_ct1.get_group(2) * ldda;

    // get norm of dx
    double lsum = 0;
    for( int j = tx; j < m; j += BLOCK_SIZE ) {
        #ifdef REAL
            double re = dx[j];
            lsum += re*re;
        #else
            double re = MAGMA_Z_REAL( dx[j] );
            double im = MAGMA_Z_IMAG( dx[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[tx] = lsum;
    magma_sum_reduce<BLOCK_SIZE>(tx, sum, item_ct1);

    if (tx == 0)
        dxnorm[item_ct1.get_group(2)] = sycl::sqrt(sum[0]);
}


/******************************************************************************/
void
magmablas_dznrm2_check_kernel(
    int m,
    magmaDoubleComplex *dA, int ldda,
    double *dxnorm, 
    double *lsticc , sycl::nd_item<3> item_ct1, double *sum)
{
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex *dx = dA + item_ct1.get_group(2) * ldda;

    // get norm of dx only if lsticc[blockIdx+1] != 0
    if (lsticc[item_ct1.get_group(2) + 1] == 0)
        return;

    double lsum = 0;
    for( int j = tx; j < m; j += BLOCK_SIZE ) {
        #ifdef REAL
            double re = dx[j];
            lsum += re*re;
        #else
            double re = MAGMA_Z_REAL( dx[j] );
            double im = MAGMA_Z_IMAG( dx[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[tx] = lsum;
    magma_sum_reduce<BLOCK_SIZE>(tx, sum, item_ct1);

    if (tx == 0)
        dxnorm[item_ct1.get_group(2)] = sycl::sqrt(sum[0]);
}


/******************************************************************************/
extern "C" void
magmablas_dznrm2_check(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc,
    magma_queue_t queue ) 
{
    sycl::range<3> threads(1, 1, BLOCK_SIZE);
    sycl::range<3> blocks(1, 1, n);
    /*
    DPCT1049:158: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magmablas_dznrm2_check_kernel(
                                 m, dA, ldda, dxnorm, dlsticc, item_ct1,
                                 sum_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
void
magmablas_dznrm2_smkernel(
    int m, int n,
    magmaDoubleComplex *dA, int ldda,
    double *dxnorm , sycl::nd_item<3> item_ct1,
    sycl::accessor<double, 2, sycl::access_mode::read_write, sycl::access::target::local> sum)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);

    for( int k = ty; k < n; k += BLOCK_SIZEy ) {
        magmaDoubleComplex *dx = dA + k * ldda;

        // get norm of dx
        double lsum = 0;
        for( int j = tx; j < m; j += BLOCK_SIZEx ) {
            #ifdef REAL
                double re = dx[j];
                lsum += re*re;
            #else
                double re = MAGMA_Z_REAL( dx[j] );
                double im = MAGMA_Z_IMAG( dx[j] );
                lsum += re*re + im*im;
            #endif
        }
        sum[tx][ty] = lsum;
        magma_sum_reduce_2d<BLOCK_SIZEx, BLOCK_SIZEy + 1>(tx, ty, sum,
                                                          item_ct1);

        if (tx == 0)
            dxnorm[k] = sycl::sqrt(sum[0][ty]);
        /*
        DPCT1065:159: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}


/******************************************************************************/
/*
    Compute the dznrm2 of each column of m-by-n matrix dA.
    The resulting norms are written in the dxnorm array.
    This routine uses only one SM (block).
*/
extern "C" void
magmablas_dznrm2_sm(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, BLOCK_SIZEy, BLOCK_SIZEx);
    sycl::range<3> blocks(1, 1, 1);
    /*
    DPCT1049:160: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 2, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(
                sycl::range<2>(32 /*BLOCK_SIZEx*/, 17 /*BLOCK_SIZEy + 1*/),
                cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magmablas_dznrm2_smkernel(m, n, dA, ldda, dxnorm,
                                                       item_ct1, sum_acc_ct1);
                         });
    });
}


/******************************************************************************/
SYCL_EXTERNAL void magma_dznrm2_adjust_kernel(double *xnorm,
                                              magmaDoubleComplex *c,
                                              sycl::nd_item<3> item_ct1,
                                              double *sum)
{
    const int tx = item_ct1.get_local_id(2);

    double temp;

    temp = MAGMA_Z_ABS( c[tx] ) / xnorm[0];
    sum[tx] = -temp * temp;
    magma_sum_reduce_n(item_ct1.get_local_range(2), tx, sum, item_ct1);

    /*
    DPCT1065:161: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tx == 0)
        xnorm[0] = xnorm[0] * sycl::sqrt(1 + sum[0]);
}


/******************************************************************************/
/*
    Adjust the norm of c to give the norm of c[k+1:], assuming that
    c was changed with orthogonal transformations.
*/
extern "C" void
magmablas_dznrm2_adjust(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDoubleComplex_ptr dc,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, k);
    sycl::range<3> blocks(1, 1, 1);
    /*
    DPCT1049:162: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_dznrm2_adjust_kernel(
                                 dxnorm, dc, item_ct1,
                                 sum_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/

#define BS 256

void
magma_dznrm2_row_check_adjust_kernel(
    int n, double tol, double *xnorm, double *xnorm2, 
    magmaDoubleComplex *C, int ldc, double *lsticc, sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2) + item_ct1.get_group(2) * BS;
    lsticc[tx+1] = 0;

    if (tx < n) {
        double temp = MAGMA_Z_ABS( C[tx*ldc] ) / xnorm[tx];
        temp = max( 0.0, ((1.0 + temp) * (1.0 - temp)) );
        
        double temp2 = xnorm[tx] / xnorm2[tx];
        temp2 = temp * (temp2 * temp2);
        
        // todo: check this accuracy procedure; currently is not working for
        //       constant matrix so it is disabled for now
        /*
        if (temp2 <= tol) {
            lsticc[tx+1] = 1;
        } else {
            xnorm[tx] *= sqrt(temp);
        }
        */
        xnorm[tx] *= sycl::sqrt(temp);
    }
    if (tx == 0)
        lsticc[0] = 0;
    magma_sum_reduce_n(item_ct1.get_local_range(2), tx, lsticc, item_ct1);
}


/******************************************************************************/
/*
    Adjust the norm of C[,1:k] to give the norm of C[k+1:,1:k], assuming that
    C was changed with orthogonal transformations.
    It also do checks for QP3
*/
extern "C" void
magmablas_dznrm2_row_check_adjust(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2, 
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, BS);
    sycl::range<3> blocks(1, 1, magma_ceildiv(k, BS));
    /*
    DPCT1049:163: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_dznrm2_row_check_adjust_kernel(
                               k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc,
                               item_ct1);
                       });
}


/******************************************************************************/
/*
    Compute the dznrm2 of each column of m-by-n matrix dA.
    The resulting norms are written in the dxnorm array. 
    The computation can be done using n blocks (default) or on one SM (commented).
*/
extern "C" void
magmablas_dznrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magma_queue_t queue ) 
{
    sycl::range<3> threads(1, 1, BLOCK_SIZE);
    sycl::range<3> blocks(1, 1, n);
    /*
    DPCT1049:164: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magmablas_dznrm2_kernel(m, dA, ldda, dxnorm,
                                                     item_ct1,
                                                     sum_acc_ct1.get_pointer());
                         });
    });

    // The following would do the computation on one SM
    // magmablas_dznrm2_sm( m, n, dA, ldda, dxnorm, queue );
}
