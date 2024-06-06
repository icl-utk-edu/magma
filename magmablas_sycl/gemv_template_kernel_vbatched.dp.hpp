/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#ifndef GEMV_TEMPLATE_KERNEL_VBATCHED_CUH
#define GEMV_TEMPLATE_KERNEL_VBATCHED_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp" // use make_FloatingPoint
#include "gemv_template_device.dp.hpp"

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
gemvn_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array, magma_int_t* incx,
    T beta, T**  y_array, magma_int_t* incy, sycl::nd_item<3> item_ct1,
    T *sdata)
{
    int batchid = item_ct1.get_group(0);

    int my_m = (int)m[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_m, TILE_SIZE)) return;

    gemvn_template_device<T, DIM_X, DIM_Y, TILE_SIZE>(
        my_m, (int)n[batchid], alpha, A_array[batchid], (int)lda[batchid],
        x_array[batchid], (int)incx[batchid], beta, y_array[batchid],
        (int)incy[batchid], item_ct1, sdata);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvn_template_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dx_array, magma_int_t* incx,
    T beta, T** dy_array, magma_int_t* incy,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, DIM_Y, DIM_X);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_m, TILE_SIZE));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        gemvn_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE>(
                            m + i, n + i, alpha, dA_array + i, ldda + i,
                            dx_array + i, incx + i, beta, dy_array + i,
                            incy + i, item_ct1, sdata_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
void
gemvc_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array,  magma_int_t* incx,
    T beta, T**  y_array, magma_int_t* incy, sycl::nd_item<3> item_ct1,
    T *sdata)
{
    int batchid = item_ct1.get_group(0);

    int my_n = (int)n[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_n, TILE_SIZE)) return;

    gemvc_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans>(
        (int)m[batchid], (int)n[batchid], alpha, A_array[batchid],
        (int)lda[batchid], x_array[batchid], (int)incx[batchid], beta,
        y_array[batchid], (int)incy[batchid], item_ct1, sdata);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_template_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dx_array, magma_int_t* incx,
    T beta, T** dy_array, magma_int_t* incy,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, DIM_Y, DIM_X);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_n, TILE_SIZE));

        if (trans == MagmaConjTrans) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<T, 1>
                        sdata_acc_ct1(sycl::range<1>(DIM_X * DIM_Y), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE,
                                                  MagmaConjTrans>(
                                m + i, n + i, alpha, dA_array + i, ldda + i,
                                dx_array + i, incx + i, beta, dy_array + i,
                                incy + i, item_ct1,
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
                            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE,
                                                  MagmaTrans>(
                                m + i, n + i, alpha, dA_array + i, ldda + i,
                                dx_array + i, incx + i, beta, dy_array + i,
                                incy + i, item_ct1,
                                sdata_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        }
    }
}

#endif  // GEMV_TEMPLATE_KERNEL_VBATCHED_CUH
