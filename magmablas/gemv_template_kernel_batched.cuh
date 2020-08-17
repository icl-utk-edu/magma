/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#ifndef GEMV_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMV_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.cuh" // use make_FloatingPoint
#include "gemv_template_device.cuh"


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
__global__ void
gemvn_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, int lda,
    T const * const * x_array,  int incx,
    T beta, T**  y_array, int incy)
{
    int batchid = blockIdx.z;

    gemvn_template_device<T, DIM_X, DIM_Y, TILE_SIZE>
        (m, n, alpha, A_array[batchid], lda, x_array[batchid], incx, beta, y_array[batchid], incy);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvn_template_batched(
    magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dx_array, magma_int_t incx,
    T beta, T** dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(m, TILE_SIZE), 1, ibatch );

        gemvn_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( m, n, alpha, dA_array+i, ldda, dx_array+i, incx, beta, dy_array+i, incy );
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
__global__ void
gemvc_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, int lda,
    T const * const * x_array,  int incx,
    T beta, T**  y_array, int incy)
{
    int batchid = blockIdx.z;

    gemvc_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans>
        (m, n, alpha, A_array[batchid], lda, x_array[batchid], incx, beta, y_array[batchid], incy);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_template_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dx_array, magma_int_t incx,
    T beta, T** dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y );

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(n, TILE_SIZE), 1, ibatch );

        if (trans == MagmaConjTrans) {
            gemvc_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, alpha, dA_array+i, ldda, dx_array+i, incx, beta, dy_array+i, incy );
        }
        else if (trans == MagmaTrans) {
            gemvc_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, alpha, dA_array+i, ldda, dx_array+i, incx, beta, dy_array+i, incy );
        }
    }
}

#endif
