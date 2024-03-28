/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.dp.hpp"
#include "zgetf2_devicefunc.dp.hpp"

#define PRECISION_z

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

/******************************************************************************/
void
izamax_kernel_batched(
        int length, magmaDoubleComplex **x_array, int xi, int xj, int lda, int incx,
        magma_int_t** ipiv_array, int ipiv_i,
        magma_int_t *info_array, int step, int gbstep,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto sdata = (double *)dpct_local;
    const int batchid = item_ct1.get_group(2);

    int tx = item_ct1.get_local_id(2);
    const magmaDoubleComplex *x = x_array[batchid] + xj * lda + xi;
    magma_int_t *ipiv           = ipiv_array[batchid] + ipiv_i;
    magma_int_t *info = &info_array[batchid];
    int linfo = ( (gbstep+step) == 0) ? 0 : *info;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(length, x, incx, shared_x, shared_idx, item_ct1);

    if (tx == 0) {
        *ipiv  = shared_idx[0] + step + 1; // Fortran Indexing & adjust ipiv
        linfo  = ( shared_x[0] == MAGMA_D_ZERO && linfo == 0) ? (shared_idx[0]+step+gbstep+1) : linfo;
        *info = (magma_int_t)linfo;
    }
}


/******************************************************************************/
void
izamax_kernel_native(
        int length, magmaDoubleComplex_ptr x, int incx,
        magma_int_t* ipiv, magma_int_t *info,
        int step, int gbstep, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto sdata = (double *)dpct_local;
    const int tx = item_ct1.get_local_id(2);

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);
    int linfo = ( (gbstep+step) == 0) ? 0 : *info;

    izamax_devfunc(length, x, incx, shared_x, shared_idx, item_ct1);
    if (tx == 0) {
        *ipiv  = shared_idx[0] + step + 1; // Fortran Indexing
        linfo  = ( shared_x[0] == MAGMA_D_ZERO && linfo == 0) ? (shared_idx[0]+step+gbstep+1) : linfo;
        *info = (magma_int_t)linfo;
    }
}


/***************************************************************************//**
    Purpose
    -------

    IZAMAX find the index of max absolute value of elements in x and store the index in ipiv

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    length       INTEGER
            On entry, length specifies the size of vector x. length >= 0.


    @param[in]
    x_array     Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension


    @param[in]
    xi      INTEGER
            Row offset, internal use

    @param[in]
    xj      INTEGER
            Column offset, internal use

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            the offset of ipiv

    @param[in]
    lda    INTEGER
            The leading dimension of each array A, internal use to find the starting position of x.

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep    INTEGER
            the offset of info, internal use

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_iamax_batched
*******************************************************************************/
extern "C" magma_int_t
magma_izamax_batched(
        magma_int_t length,
        magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t lda, magma_int_t incx,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t step, magma_int_t gbstep, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, zamax);

    int chunk = magma_ceildiv( length, zamax );

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:1662: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(
                sycl::range<1>(zamax * (sizeof(double) + sizeof(int))), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             izamax_kernel_batched(
                                 length, x_array, xi, xj, lda, incx, ipiv_array,
                                 ipiv_i, info_array, step, gbstep, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
    });

    return 0;
}


/******************************************************************************/
// For use in magma_izamax_native only
// cublasIzamax always writes 32bit pivots, so make sure it is magma_int_t
void magma_zpivcast(magma_int_t* dipiv)
{
    // uses only 1 thread
    int* address = (int*)dipiv;
    int pivot = *address;          // read the value written by cuBLAS (int)
    *dipiv = (magma_int_t)pivot;    // write it back in the same address as dipiv
}

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_native(
    magma_int_t length,
    magmaDoubleComplex_ptr x, magma_int_t incx,
    magma_int_t* ipiv, magma_int_t *info,
    magma_int_t step, magma_int_t gbstep, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    // TODO: decide the best izamax for all precisions
    if( length <= 15360 ) {
        sycl::range<3> grid(1, 1, 1);
        sycl::range<3> threads(1, 1, zamax);

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                /*
                DPCT1083:1663: The size of local memory in the migrated code may
                be different from the original code. Check that the allocated
                memory size in the migrated code is correct.
                */
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(
                        sycl::range<1>(zamax * (sizeof(double) + sizeof(int))),
                        cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     izamax_kernel_native(
                                         length, x, incx, ipiv, info, step,
                                         gbstep, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    else {
        int ptr_mode;
        /*
        DPCT1026:586: The call to cublasGetPointerMode was removed because the
        function call is redundant in DPC++.
        */
        /*
        DPCT1026:587: The call to cublasSetPointerMode was removed because the
        function call is redundant in DPC++.
        */

        int64_t *res_temp_ptr_ct1 =
            sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
        oneapi::mkl::blas::column_major::iamax(*queue->syclblas_handle(), length,
                                               (std::complex<double> *)x, 1,
                                               res_temp_ptr_ct1)
            .wait();
        int res_temp_host_ct2 = (int)*res_temp_ptr_ct1;
        dpct::dpct_memcpy((int *)(ipiv), &res_temp_host_ct2, sizeof(int));
        sycl::free(res_temp_ptr_ct1, dpct::get_default_queue());
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
                                             sycl::range<3>(1, 1, 1)),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_zpivcast(ipiv);
                           });

        adjust_ipiv( ipiv, 1, step, queue);
    }
    return 0;
}

/******************************************************************************/

void zswap_kernel_batched(
        magma_int_t n,
        magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx,
        magma_int_t step, magma_int_t** ipiv_array, sycl::nd_item<3> item_ct1,
        int *jp)
{
    const int batchid = item_ct1.get_group(2);
    magmaDoubleComplex *x = x_array[batchid] + xj * incx + xi;
    magma_int_t *ipiv = ipiv_array[batchid] + xi;

    magmaDoubleComplex* xpiv = x_array[batchid] + (xj+step) * incx + (xi+step);
    double rx_abs = fabs( MAGMA_Z_REAL(xpiv[0]) ) + fabs( MAGMA_Z_IMAG(xpiv[0]) );
    if( rx_abs != MAGMA_D_ZERO) {
        zswap_device(n, x, incx, step, ipiv, item_ct1, jp);
    }
}


/******************************************************************************/

void zswap_kernel_native( magma_int_t n,
                          magmaDoubleComplex_ptr x, magma_int_t incx,
                          magma_int_t step, magma_int_t* ipiv,
                          sycl::nd_item<3> item_ct1, int *jp)
{
    zswap_device(n, x, incx, step, ipiv, item_ct1, jp);
}


/***************************************************************************//**
    Purpose
    -------

    zswap two row in x.  index (ipiv[step]-1)-th and index step -th

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            On entry, n specifies the size of vector x. n >= 0.


    @param[in]
    dA_array  Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension


    @param[in]
    ai      INTEGER
            Row offset, internal use.

    @param[in]
    aj      INTEGER
            Column offset, internal use.

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swap_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zswap_batched( magma_int_t n,
                     magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t incx,
                     magma_int_t step, magma_int_t** ipiv_array,
                     magma_int_t batchCount, magma_queue_t queue)
{
    /*
    zswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s nb=%lld > %lld, not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, zamax);

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 0>
            jp_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zswap_kernel_batched(n, dA_array, ai, aj, incx,
                                                  step, ipiv_array, item_ct1,
                                                  jp_acc_ct1.get_pointer());
                         });
    });
    return 0;
}


/******************************************************************************/
extern "C" void
magma_zswap_native( magma_int_t n, magmaDoubleComplex_ptr x, magma_int_t incx,
                    magma_int_t step, magma_int_t* ipiv,
                    magma_queue_t queue)
{
    /*
    zswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS){
        fprintf( stderr, "%s nb=%lld > %lld, not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
    }
    sycl::range<3> grid(1, 1, 1);
    sycl::range<3> threads(1, 1, zamax);

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 0>
            jp_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zswap_kernel_native(n, x, incx, step, ipiv,
                                                 item_ct1,
                                                 jp_acc_ct1.get_pointer());
                         });
    });
}

/******************************************************************************/
template<int N>

void zscal_zgeru_1d_kernel_native( int m,
                                magmaDoubleComplex_ptr dA, int lda,
                                magma_int_t *info, int step, int gbstep,
                                sycl::nd_item<3> item_ct1,
                                magmaDoubleComplex *shared_y)
{
    // This dev function has a return statement inside, be sure
    // not to merge it with another dev function. Otherwise, the
    // return statement should be converted into an if-statement
    zscal_zgeru_device<N>(m, dA, lda, info, step, gbstep, item_ct1, shared_y);
}


/******************************************************************************/

void zscal_zgeru_1d_generic_kernel_native( int m, int n,
                                magmaDoubleComplex_ptr dA, int lda,
                                magma_int_t *info, int step, int gbstep,
                                sycl::nd_item<3> item_ct1)
{
    // This dev function has a return statement inside, be sure
    // not to merge it with another dev function. Otherwise, the
    // return statement should be converted into an if-statement
    zscal_zgeru_generic_device(m, n, dA, lda, info, step, gbstep, item_ct1);
}


/******************************************************************************/
template<int N>

void zscal_zgeru_1d_kernel_batched(int m, magmaDoubleComplex **dA_array, int ai, int aj, int lda, magma_int_t *info_array, int step, int gbstep,
                                   sycl::nd_item<3> item_ct1,
                                   magmaDoubleComplex *shared_y)
{
    const int batchid = item_ct1.get_group(0);
    magmaDoubleComplex* dA = dA_array[batchid] + aj * lda + ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_device<N>(m, dA, lda, info, step, gbstep, item_ct1, shared_y);
}


/******************************************************************************/

void zscal_zgeru_1d_generic_kernel_batched(int m, int n, magmaDoubleComplex **dA_array, int ai, int aj, int lda, magma_int_t *info_array, int step, int gbstep,
                                           sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    magmaDoubleComplex* dA = dA_array[batchid] + aj * lda + ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_generic_device(m, n, dA, lda, info, step, gbstep, item_ct1);
}


/******************************************************************************/
extern "C"
magma_int_t
magma_zscal_zgeru_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) {
        fprintf( stderr, "%s nb=%lld, > %lld, not supported\n", __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }

    magma_int_t max_batchCount = queue->get_maxBatch();
    const int tbx = 256;
    sycl::range<3> threads(1, 1, tbx);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, tbx));

        switch(n){
            case 1: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(1), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<1>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:592: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 2: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(2), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<2>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:593: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 3: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(3), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<3>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:594: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 4: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(4), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<4>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:595: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 5: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(5), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<5>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:596: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 6: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(6), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<6>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:597: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 7: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(7), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<7>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:598: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 8: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        shared_y_acc_ct1(sycl::range<1>(8), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zscal_zgeru_1d_kernel_batched<8>(
                                             m, dA_array + i, ai, aj, lda,
                                             info_array + i, step, gbstep,
                                             item_ct1,
                                             shared_y_acc_ct1.get_pointer());
                                     });
                });
            break;
            /*
            DPCT1049:599: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            default: ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zscal_zgeru_1d_generic_kernel_batched(
                                       m, n, dA_array + i, ai, aj, lda,
                                       info_array + i, step, gbstep, item_ct1);
                               });
        }
    }
    return 0;
}


/******************************************************************************/
extern "C"
magma_int_t
magma_zscal_zgeru_native(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t lda,
    magma_int_t *info, magma_int_t step, magma_int_t gbstep,
    magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) {
        fprintf( stderr, "%s nb=%lld, > %lld, not supported\n", __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    const int tbx = 256;
    sycl::range<3> grid(1, 1, magma_ceildiv(m, tbx));
    sycl::range<3> threads(1, 1, tbx);
    switch(n){
        case 1: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(1), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<1>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:601: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 2: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(2), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<2>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:602: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 3: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(3), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<3>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:603: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 4: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(4), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<4>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:604: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 5: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(5), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<5>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:605: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 6: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(6), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<6>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:606: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 7: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(7), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<7>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:607: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    shared_y_acc_ct1(sycl::range<1>(8), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zscal_zgeru_1d_kernel_native<8>(
                                         m, dA, lda, info, step, gbstep,
                                         item_ct1,
                                         shared_y_acc_ct1.get_pointer());
                                 });
            });
        break;
        /*
        DPCT1049:608: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        default: ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zscal_zgeru_1d_generic_kernel_native(
                                   m, n, dA, lda, info, step, gbstep, item_ct1);
                           });
    }
    return 0;
}


/******************************************************************************/

void zgetf2trsm_kernel_batched(int ib, int n, magmaDoubleComplex **dA_array, int step, int lda,
                               sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{

    auto shared_data = (magmaDoubleComplex *)dpct_local;

    /*
        this kernel does the safe nonblocked TRSM operation
        B = A^-1 * B
    */
    const int batchid = item_ct1.get_group(2);

    magmaDoubleComplex *A_start = dA_array[batchid];
    magmaDoubleComplex *A = &(A_start[step + step * lda]);
    magmaDoubleComplex *B = &(A_start[step + (step+ib) * lda]);
    magmaDoubleComplex *shared_a = shared_data;
    magmaDoubleComplex *shared_b = shared_data+ib*ib;

    int tid = item_ct1.get_local_id(2);
    int i,d;


    // Read A and B at the same time to the shared memory (shared_a shared_b)
    // note that shared_b = shared_a+ib*ib so its contiguous
    // I can make it in one loop reading
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n+ib; i++) {
            shared_a[tid + i*ib] = A[tid + i*lda];
        }
    }
    /*
    DPCT1065:609: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (tid < n) {
        #pragma unroll
        for (d=0;  d < ib-1; d++) {
            for (i=d+1; i < ib; i++) {
                /*
                DPCT1064:611: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                shared_b[i + tid * ib] +=
                    (MAGMA_Z_NEG_ONE)*shared_a[i + d * ib] *
                    shared_b[d + tid * ib];
            }
        }
    }
    /*
    DPCT1065:610: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write back B
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n; i++) {
            B[tid + i*lda] = shared_b[tid + i*ib];
        }
    }
}


/***************************************************************************//**
    Purpose
    -------

    zgetf2trsm solves one of the matrix equations on gpu

     B = C^-1 * B

    where C, B are part of the matrix A in dA_array,

    This version load C, B into shared memory and solve it
    and copy back to GPU device memory.
    This is an internal routine that might have many assumption.

    Arguments
    ---------
    @param[in]
    ib       INTEGER
            The number of rows/columns of each matrix C, and rows of B.  ib >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix B.  n >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" void
magma_zgetf2trsm_batched(magma_int_t ib, magma_int_t n, magmaDoubleComplex **dA_array,
                         magma_int_t step, magma_int_t ldda,
                         magma_int_t batchCount, magma_queue_t queue)
{
    if ( n == 0 || ib == 0 ) return;
    /*
    DPCT1083:613: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_size = sizeof(magmaDoubleComplex) * (ib * (ib + n));

    // TODO TODO TODO
    if ( shared_size > (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 46K leaving 2K for extra
    {
        fprintf( stderr, "%s: error out of shared memory\n", __func__ );
        return;
    }

    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, max(n, ib));

    /*
    DPCT1049:612: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(shared_size), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zgetf2trsm_kernel_batched(
                                 ib, n, dA_array, step, ldda, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
template<int NB>
void
zgetf2trsm_2d_kernel( int m, int n,
                           magmaDoubleComplex_ptr dA, int ldda,
                           magmaDoubleComplex_ptr dB, int lddb,
                           sycl::nd_item<3> item_ct1, magmaDoubleComplex *sA,
                           magmaDoubleComplex *sB)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);

    // init sA & sB
    /*
    DPCT1064:615: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sA[ty * NB + tx] = MAGMA_Z_ZERO;
    /*
    DPCT1064:616: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    sB[ty * NB + tx] = MAGMA_Z_ZERO;

    const int nblocks = magma_ceildiv(n, NB);
    const int n_ = n - (nblocks-1) * NB;

    // load A
    if( ty < m && tx < m && tx > ty){
        sA[ty * NB + tx] = dA[ty * ldda + tx];
    }

    if( ty == tx ){
        // ignore diagonal elements
        /*
        DPCT1064:617: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        sA[tx * NB + tx] = MAGMA_Z_ONE;
    }
    /*
    DPCT1065:614: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma  unroll
    for(int s = 0; s < nblocks-1; s++){
        // load B
        if( tx < m ){
            sB[ ty * NB + tx] = dB[ ty * lddb + tx ];
        }

        // no need to sync because each thread column is less than 32
        // solve
        #pragma unroll
        for(int i = 0; i < NB; i++){
            if(tx >  i){
                 sB[ ty * NB + tx ] -= sA[ i * NB + tx ] * sB[ ty * NB + i ];
            }
        }

        // write B
        if( tx < m){
            dB[ ty * lddb + tx ] = sB[ ty * NB + tx ];
        }
        dB += NB * lddb;
    }

    // last, possible partial, block
    if( ty < n_ && tx < m){
        sB[ ty * NB + tx] = dB[ ty * lddb + tx ];
    }

    #pragma unroll
    for(int i = 0; i < NB; i++){
        if(tx >  i){
             sB[ ty * NB + tx ] -= sA[ i * NB + tx ] * sB[ ty * NB + i ];
        }
    }

    if( ty < n_ && tx < m){
        dB[ ty * lddb + tx ] = sB[ ty * NB + tx ];
    }
}


/******************************************************************************/
extern"C" void
magma_zgetf2trsm_2d_native(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_queue_t queue)
{
    if( m > 32 ){
        magma_ztrsm(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            /*
            DPCT1064:618: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            m, n, MAGMA_Z_ONE, dA, ldda, dB, lddb, queue);
        return;
    }

    const int m8 = magma_roundup(m, 8);
    sycl::range<3> grid(1, 1, 1);
    sycl::range<3> threads(1, m8, m8);

    switch(m8){
        /*
        DPCT1049:619: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sA_acc_ct1(sycl::range<1>(8 * 8), cgh);
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sB_acc_ct1(sycl::range<1>(8 * 8), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2trsm_2d_kernel<8>(
                                         m, n, dA, ldda, dB, lddb, item_ct1,
                                         sA_acc_ct1.get_pointer(),
                                         sB_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:620: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 16: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sA_acc_ct1(sycl::range<1>(16 * 16), cgh);
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sB_acc_ct1(sycl::range<1>(16 * 16), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2trsm_2d_kernel<16>(
                                         m, n, dA, ldda, dB, lddb, item_ct1,
                                         sA_acc_ct1.get_pointer(),
                                         sB_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:621: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 24: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sA_acc_ct1(sycl::range<1>(24 * 24), cgh);
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sB_acc_ct1(sycl::range<1>(24 * 24), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2trsm_2d_kernel<24>(
                                         m, n, dA, ldda, dB, lddb, item_ct1,
                                         sA_acc_ct1.get_pointer(),
                                         sB_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:622: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 32: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sA_acc_ct1(sycl::range<1>(32 * 32), cgh);
                sycl::local_accessor<magmaDoubleComplex, 1>
                    sB_acc_ct1(sycl::range<1>(32 * 32), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgetf2trsm_2d_kernel<32>(
                                         m, n, dA, ldda, dB, lddb, item_ct1,
                                         sA_acc_ct1.get_pointer(),
                                         sB_acc_ct1.get_pointer());
                                 });
            });
            break;
        default:;
    }
}

/******************************************************************************/
void
zcomputecolumn_kernel_shared_batched( int m, int paneloffset, int step,
                                      magmaDoubleComplex **dA_array, int ai, int aj,
                                      int lda, magma_int_t **ipiv_array, magma_int_t *info_array, int gbstep,
                                      sycl::nd_item<3> item_ct1,
                                      uint8_t *dpct_local, double *shared_x,
                                      int *shared_idx, magmaDoubleComplex *alpha)
{
    const int batchid = item_ct1.get_group(2);
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    int gboff = paneloffset+step;
    magma_int_t *ipiv           = ipiv_array[batchid] + ai;
    magmaDoubleComplex *A_start = dA_array[batchid] + aj * lda + ai;
    magmaDoubleComplex *A0j     = &(A_start[paneloffset + (paneloffset+step) * lda]);
    magmaDoubleComplex *A00     = &(A_start[paneloffset + paneloffset * lda]);

    magmaDoubleComplex *shared_A = shared_data;

    int tid = item_ct1.get_local_id(2);
    int linfo = ((gboff + gbstep) == 0 ) ? 0 : info_array[batchid];

    // checkinfo to avoid computation of the singular matrix
//    if (info_array[batchid] != 0 ) return;


    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    // read the current column from dev to shared memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) shared_A[tid + s * MAX_NTHREADS] = A0j[tid + s * MAX_NTHREADS];
    }
    /*
    DPCT1065:623: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // update this column
    if ( step > 0 ) {
        zupdate_device(m, step, A00, lda, shared_A, 1, item_ct1);
        /*
        DPCT1065:628: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // if ( tid < (m-step) ) // DO NO TPUT THE IF CONDITION HERE SINCE izamax_devfunc HAS __syncthreads INSIDE.
    // So let all htreads call this routine it will handle correctly based on the size
    // note that izamax need only 128 threads, s
    izamax_devfunc(m - step, shared_A + step, 1, shared_x, shared_idx,
                   item_ct1);
    if (tid == 0) {
        ipiv[gboff]  = shared_idx[0] + gboff + 1; // Fortran Indexing
        *alpha = shared_A[shared_idx[0] + step];
        //printf("@ step %d ipiv=%d where gboff=%d  shared_idx %d alpha %5.3f\n",step,ipiv[gboff],gboff,shared_idx[0],alpha);
        linfo  = ( shared_x[0] == MAGMA_D_ZERO && linfo == 0) ? (shared_idx[0]+gboff+gbstep+1) : linfo;
        info_array[batchid] = (magma_int_t)linfo;
    }
    /*
    DPCT1065:624: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
//    if (shared_x[0] == MAGMA_D_ZERO) return;
    /*
    DPCT1065:625: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
//    item_ct1.barrier();

    if( shared_x[0] != MAGMA_D_ZERO ) {
        zscal5_device( m-step, shared_A+step, *alpha, item_ct1);
        // there is sync at the end of zscal5_device
    }

    // put back the pivot that has been scaled with itself menaing =1
    if (tid == 0) shared_A[shared_idx[0] + step] = *alpha;
    /*
    DPCT1065:626: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write back from shared to dev memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m )
        {
            A0j[tid + s * MAX_NTHREADS] = shared_A[tid + s * MAX_NTHREADS];
            //printf("@ step %d tid %d updating A=x*alpha after A= %5.3f\n",step,tid,shared_A[tid]);
        }
    }
    /*
    DPCT1065:627: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/******************************************************************************/
extern "C"
magma_int_t magma_zcomputecolumn_batched( magma_int_t m, magma_int_t paneloffset, magma_int_t step,
                                          magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
                                          magma_int_t **ipiv_array,
                                          magma_int_t *info_array, magma_int_t gbstep,
                                          magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( m == 0) return 0;

    size_t all_shmem_size = zamax*(sizeof(double)+sizeof(int)) + (m+2)*sizeof(magmaDoubleComplex);
    if ( all_shmem_size >  (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 44K leaving 4K for extra
    {
        fprintf( stderr, "%s error out of shared memory\n", __func__ );
        return -20;
    }

    /*
    DPCT1083:630: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_size = sizeof(magmaDoubleComplex) * m;
    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, min(m, MAX_NTHREADS));

    /*
    DPCT1049:629: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1>
            dpct_local_acc_ct1(sycl::range<1>(shared_size), cgh);
        sycl::local_accessor<double, 1>
            shared_x_acc_ct1(sycl::range<1>(zamax), cgh);
        sycl::local_accessor<int, 1>
            shared_idx_acc_ct1(sycl::range<1>(zamax), cgh);
        sycl::local_accessor<magmaDoubleComplex, 0>
            alpha_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zcomputecolumn_kernel_shared_batched(
                                 m, paneloffset, step, dA_array, ai, aj, lda,
                                 ipiv_array, info_array, gbstep, item_ct1,
                                 dpct_local_acc_ct1.get_pointer(),
                                 shared_x_acc_ct1.get_pointer(),
                                 shared_idx_acc_ct1.get_pointer(),
                                 alpha_acc_ct1.get_pointer());
                         });
    });

    return 0;
}

/******************************************************************************/
template<int N>
void
zgetf2_fused_kernel_batched( int m,
                           magmaDoubleComplex** dA_array, int ai, int aj, int ldda,
                           magma_int_t** dipiv_array, magma_int_t* info_array, int batchCount,
                           sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int tx = item_ct1.get_local_id(2);
    const int batchid = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                        item_ct1.get_local_id(1);
    if(batchid >= batchCount)return;

    int rowid, gbstep = aj;
    int linfo = (gbstep == 0) ? 0 : info_array[batchid];

    // shared memory workspace
    auto zdata = (magmaDoubleComplex *)dpct_local;
    magmaDoubleComplex* swork = (magmaDoubleComplex*)zdata;

    // read
    magmaDoubleComplex* dA = dA_array[batchid] + aj * ldda + ai;
    /*
    DPCT1064:631: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
#pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }

     zgetf2_fused_device<N>(m, min(m, N), rA, dipiv_array[batchid] + ai, swork,
                            linfo, gbstep, rowid, item_ct1);

    // write
    if(tx == 0){
        info_array[batchid] = (magma_int_t)( linfo );
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        dA[ i * ldda + rowid ] = rA[i];
    }

}

/******************************************************************************/
template <int N>
static magma_int_t magma_zgetf2_fused_kernel_driver_batched(
    magma_int_t m, magmaDoubleComplex **dA_array, magma_int_t ai,
    magma_int_t aj, magma_int_t ldda, magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue) try {
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    magma_int_t ntcol = (m >= 32)? 1 : (32/m);
    int shmem = 0, shmem_max = 0;   // not magma_int_t (causes problems with 64bit builds)
    shmem += N * sizeof(magmaDoubleComplex);
    shmem += m * sizeof(double);
    shmem += m * sizeof(int);    // not magma_int_t
    shmem += N * sizeof(int);    // not magma_int_t
    shmem *= ntcol;

    sycl::range<3> grid(1, 1, magma_ceildiv(batchCount, ntcol));
    sycl::range<3> threads(1, ntcol, m);

    // get max. dynamic shared memory on the GPU
    int nthreads_max;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();

    int nthreads = m * ntcol;
    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

//    void *kernel_args[] = {&m, &dA_array, &ai, &aj, &ldda, &dipiv_array, &info_array, &batchCount};
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
       sycl::local_accessor<uint8_t, 1>
                       dpct_local_acc_ct1(sycl::range<1>(shmem), cgh); // NNB: I added this manually, dpct didn't finish --
				                                      // check if size is correct
      cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                       zgetf2_fused_kernel_batched<N>(m, dA_array, ai, aj, ldda, dipiv_array, info_array,
                            batchCount, item_ct1, dpct_local_acc_ct1.get_pointer());
                       });
	});
    /*
    DPCT1000:633: Error handling if-stmt was detected but could not be
    rewritten.
    */
    // TODO
//    if (e != 0) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        /*
        DPCT1001:632: The statement could not be removed.
        */
//        arginfo = -100;
//    }

    return arginfo;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/***************************************************************************//**
    Purpose
    -------
    magma_zgetf2_reg_batched computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. This routine is used for batch LU panel
    factorization, and has specific assumption about the value of N

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is a right-looking unblocked version of the algorithm. The routine is a batched
    version that factors batchCount M-by-N matrices in parallel.

    This version load an entire matrix (m*n) into registers and factorize it with pivoting
    and copy back to GPU device memory.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  ib >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_fused_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if(m < 0) {
        info = -1;
    }
    else if(n < 0 || n > 32){
        fprintf( stderr, "%s: n = %4lld not supported, must be between 0 and %4lld\n",
                 __func__, (long long) m, (long long) 32);
        info = -2;
    }

    if(info < 0) return info;

    switch(n) {
        case  1: info = magma_zgetf2_fused_kernel_driver_batched< 1>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  2: info = magma_zgetf2_fused_kernel_driver_batched< 2>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  3: info = magma_zgetf2_fused_kernel_driver_batched< 3>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  4: info = magma_zgetf2_fused_kernel_driver_batched< 4>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  5: info = magma_zgetf2_fused_kernel_driver_batched< 5>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  6: info = magma_zgetf2_fused_kernel_driver_batched< 6>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  7: info = magma_zgetf2_fused_kernel_driver_batched< 7>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  8: info = magma_zgetf2_fused_kernel_driver_batched< 8>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  9: info = magma_zgetf2_fused_kernel_driver_batched< 9>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 10: info = magma_zgetf2_fused_kernel_driver_batched<10>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 11: info = magma_zgetf2_fused_kernel_driver_batched<11>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 12: info = magma_zgetf2_fused_kernel_driver_batched<12>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 13: info = magma_zgetf2_fused_kernel_driver_batched<13>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 14: info = magma_zgetf2_fused_kernel_driver_batched<14>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 15: info = magma_zgetf2_fused_kernel_driver_batched<15>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 16: info = magma_zgetf2_fused_kernel_driver_batched<16>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 17: info = magma_zgetf2_fused_kernel_driver_batched<17>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 18: info = magma_zgetf2_fused_kernel_driver_batched<18>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 19: info = magma_zgetf2_fused_kernel_driver_batched<19>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 20: info = magma_zgetf2_fused_kernel_driver_batched<20>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 21: info = magma_zgetf2_fused_kernel_driver_batched<21>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 22: info = magma_zgetf2_fused_kernel_driver_batched<22>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 23: info = magma_zgetf2_fused_kernel_driver_batched<23>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 24: info = magma_zgetf2_fused_kernel_driver_batched<24>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 25: info = magma_zgetf2_fused_kernel_driver_batched<25>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 26: info = magma_zgetf2_fused_kernel_driver_batched<26>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 27: info = magma_zgetf2_fused_kernel_driver_batched<27>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 28: info = magma_zgetf2_fused_kernel_driver_batched<28>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 29: info = magma_zgetf2_fused_kernel_driver_batched<29>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 30: info = magma_zgetf2_fused_kernel_driver_batched<30>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 31: info = magma_zgetf2_fused_kernel_driver_batched<31>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 32: info = magma_zgetf2_fused_kernel_driver_batched<32>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        default: info = -100;
    }

    return info;
}
