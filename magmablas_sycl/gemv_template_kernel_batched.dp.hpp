/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/
#ifndef GEMV_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMV_TEMPLATE_KERNEL_BATCHED_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp" // use make_FloatingPoint
#include "gemv_template_device.dp.hpp"

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
gemvn_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, T const * A, int lda, int strideA,
    T const * const * x_array, T const * x, int incx, int stridex,
    T beta, T**  y_array, T* y, int incy, int stridey ,
    sycl::nd_item<3> item_ct1, T *sdata)
{
    const int batchid = item_ct1.get_group(0);
    const T* dA = (A_array == NULL) ? (A + batchid * strideA) : A_array[batchid];
    const T* dx = (x_array == NULL) ? (x + batchid * stridex) : x_array[batchid];
    T*       dy = (y_array == NULL) ? (y + batchid * stridey) : y_array[batchid];

    gemvn_template_device<T, DIM_X, DIM_Y, TILE_SIZE>(
        m, n, alpha, dA, lda, dx, incx, beta, dy, incy, item_ct1, sdata);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvn_template_batched(
    magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex,
    T beta, T** dy_array, T* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, DIM_Y, DIM_X);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, TILE_SIZE));

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

	        cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        gemvn_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE>(
                            m, n, alpha, dA_array_i, dA + (i * strideA), ldda,
                            strideA, dx_array_i, dx + (i * stridex), incx,
                            stridex, beta, dy_array_i, dy + (i * stridey), incy,
                            stridey, item_ct1, sdata_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
void
gemvc_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, T const * A, int lda,  int strideA,
    T const * const * x_array, T const * x, int incx, int stridex,
    T beta, T**  y_array, T* y, int incy, int stridey ,
    sycl::nd_item<3> item_ct1, T *sdata)
{
    int batchid = item_ct1.get_group(0);
    const T* dA = (A_array == NULL) ? (A + batchid * strideA) : A_array[batchid];
    const T* dx = (x_array == NULL) ? (x + batchid * stridex) : x_array[batchid];
    T*       dy = (y_array == NULL) ? (y + batchid * stridey) : y_array[batchid];

    gemvc_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans>(
        m, n, alpha, dA, lda, dx, incx, beta, dy, incy, item_ct1, sdata);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_template_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex,
    T beta, T** dy_array, T* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, DIM_Y, DIM_X);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, TILE_SIZE));

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        if (trans == MagmaConjTrans) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<T, 1>
                        sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            gemvc_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE,
                                                 MagmaConjTrans>(
                                m, n, alpha, dA_array_i, dA + (i * strideA),
                                ldda, strideA, dx_array_i, dx + (i * stridex),
                                incx, stridex, beta, dy_array_i,
                                dy + (i * stridey), incy, stridey, item_ct1,
                                sdata_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        }
        else if (trans == MagmaTrans) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<T, 1>
                        sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            gemvc_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE,
                                                 MagmaTrans>(
                                m, n, alpha, dA_array_i, dA + (i * strideA),
                                ldda, strideA, dx_array_i, dx + (i * stridex),
                                incx, stridex, beta, dy_array_i,
                                dy + (i * stridey), incy, stridey, item_ct1,
                                sdata_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        }
    }
}

#endif
