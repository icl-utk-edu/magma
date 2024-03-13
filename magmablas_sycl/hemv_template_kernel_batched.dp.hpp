/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef HEMV_TEMPLATE_KERNEL_BATCHED_CUH
#define HEMV_TEMPLATE_KERNEL_BATCHED_CUH
////////////////////////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "atomics.dp.hpp"
#include "hemv_template_device.dp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static 
void hemv_diag_template_batched_kernel(
        magma_uplo_t uplo, int n,
        T alpha, T** Aarray, int ldda,
                 T** Xarray, int incx,
        T beta,  T** Yarray, int incy,
        int offA, int offX, int offY, sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    int batchid = item_ct1.get_group(0);

    hemv_diag_device<T, NB, TY>(
        uplo, n, alpha, Aarray[batchid] + offA * ldda + offA, ldda,
        Xarray[batchid] + offX * incx, incx, beta,
        Yarray[batchid] + offY * incy, incy, item_ct1, sA, sX);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static 
void hemv_lower_template_batched_kernel(
        int n, T alpha,
        T** Aarray, int ldda,
        T** Xarray, int incx,
        T** Yarray, int incy,
        int offA, int offX, int offY, sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    int batchid = item_ct1.get_group(0);

    hemv_lower_device<T, NB, TY>(n, alpha, Aarray[batchid] + offA * ldda + offA,
                                 ldda, Xarray[batchid] + offX * incx, incx,
                                 Yarray[batchid] + offY * incy, incy, item_ct1,
                                 sA, sX);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static 
void hemv_upper_template_batched_kernel(
        int n, T alpha,
        T** Aarray, int ldda,
        T** Xarray, int incx,
        T** Yarray, int incy,
        int offA, int offX, int offY, sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    int batchid = item_ct1.get_group(0);

    hemv_upper_device<T, NB, TY>(n, alpha, Aarray[batchid] + offA * ldda + offA,
                                 ldda, Xarray[batchid] + offX * incx, incx,
                                 Yarray[batchid] + offY * incy, incy, item_ct1,
                                 sA, sX);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// diag
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_diag_template_batched(
    magma_uplo_t uplo, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dX_array, magma_int_t incx,
    T beta,  T** dY_array, magma_int_t incy,
    magma_int_t offA, magma_int_t offX, magma_int_t offY,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, TY, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::local_accessor<T, 1>
                    sX_acc_ct1(sycl::range<1>(NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        hemv_diag_template_batched_kernel<T, NB, TY>(
                            uplo, n, alpha, dA_array + i, ldda, dX_array + i,
                            incx, beta, dY_array + i, incy, offA, offX, offY,
                            item_ct1, sA_acc_ct1.get_pointer(),
                            sX_acc_ct1.get_pointer());
                    });
            });
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lower
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_lower_template_batched(
    magma_int_t n, T alpha,
    T** dA_array, magma_int_t ldda,
    T** dX_array, magma_int_t incx,
    T** dY_array, magma_int_t incy,
    magma_int_t offA, magma_int_t offX, magma_int_t offY,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, TY, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * (NB + 1)), cgh);
                sycl::local_accessor<T, 1>
                    sX_acc_ct1(sycl::range<1>(NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        hemv_lower_template_batched_kernel<T, NB, TY>(
                            n, alpha, dA_array + i, ldda, dX_array + i, incx,
                            dY_array + i, incy, offA, offX, offY, item_ct1,
                            sA_acc_ct1.get_pointer(), sX_acc_ct1.get_pointer());
                    });
            });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// upper
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_upper_template_batched(
    magma_int_t n, T alpha,
    T** dA_array, magma_int_t ldda,
    T** dX_array, magma_int_t incx,
    T** dY_array, magma_int_t incy,
    magma_int_t offA, magma_int_t offX, magma_int_t offY,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, TY, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * (NB + 1)), cgh);
                sycl::local_accessor<T, 1>
                    sX_acc_ct1(sycl::range<1>(NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        hemv_upper_template_batched_kernel<T, NB, TY>(
                            n, alpha, dA_array + i, ldda, dX_array + i, incx,
                            dY_array + i, incy, offA, offX, offY, item_ct1,
                            sA_acc_ct1.get_pointer(), sX_acc_ct1.get_pointer());
                    });
            });
    }
}
#endif //HEMV_TEMPLATE_KERNEL_BATCHED_CUH
