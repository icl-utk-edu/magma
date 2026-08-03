/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       @author Natalie Beams

*/
#ifndef GEMV_PACKED_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMV_PACKED_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.cuh" // use make_FloatingPoint
#include "gemv_packed_template_device.cuh"


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, const int LOWER>
__global__ void
gemvn_packed_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, T const * A, int  lda, int strideA, int Ai, int Aj,
    T const * const * x_array, T const * x, int incx, int stridex, int xi,
    T beta, T**       y_array, T       * y, int incy, int stridey, int yi )
{
    const int batchid = blockIdx.z;
    const T* dA = (A_array == NULL) ? (A + batchid * strideA) : ( A_array[batchid] );
    const T* dx = (x_array == NULL) ? (x + batchid * stridex + xi * incx)     : ( x_array[batchid] + xi * incx );
    T*       dy = (y_array == NULL) ? (y + batchid * stridey + yi * incy)     : ( y_array[batchid] + yi * incy );

    gemvn_packed_template_device<T, DIM_X, DIM_Y, TILE_SIZE, LOWER>
    (m, n, alpha, dA, Ai, Aj, lda, dx, incx, beta, dy, incy);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvn_lower_packed_template_batched(
    magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    T beta, T**       dy_array, T       * dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(m, TILE_SIZE), 1, ibatch );

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        gemvn_packed_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, 1>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA, Ai, Aj,
                       dx_array_i, dx+(i*stridex), incx, stridex, xi,
                beta,  dy_array_i, dy+(i*stridey), incy, stridey, yi );
    }
}

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvn_upper_packed_template_batched(
    magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    T beta, T**       dy_array, T       * dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(m, TILE_SIZE), 1, ibatch );

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        gemvn_packed_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, 0>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA, Ai, Aj,
                       dx_array_i, dx+(i*stridex), incx, stridex, xi,
                beta,  dy_array_i, dy+(i*stridey), incy, stridey, yi );
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans, const int LOWER>
__global__ void
gemvc_packed_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, T const * A, int lda,  int strideA, int Ai, int Aj,
    T const * const * x_array, T const * x, int incx, int stridex, int xi,
    T beta, T**       y_array, T       * y, int incy, int stridey, int yi)
{
    int batchid = blockIdx.z;
    const T* dA = (A_array == NULL) ? (A + batchid * strideA ) : ( A_array[batchid] );
    const T* dx = (x_array == NULL) ? (x + batchid * stridex + xi * incx)     : ( x_array[batchid] + xi * incx );
    T*       dy = (y_array == NULL) ? (y + batchid * stridey + yi * incy)     : ( y_array[batchid] + yi * incy );

    gemvc_packed_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans, LOWER>
    (m, n, alpha, dA, Ai, Aj, lda, dx, incx, beta, dy, incy);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_lower_packed_template_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    T beta, T**       dy_array, T       * dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y );

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(n, TILE_SIZE), 1, ibatch );

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        if (trans == MagmaConjTrans) {
            gemvc_packed_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans, 1>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA, Ai, Aj,
                           dx_array_i, dx+(i*stridex), incx, stridex, xi,
                    beta,  dy_array_i, dy+(i*stridey), incy, stridey, yi);
        }
        else if (trans == MagmaTrans) {
            gemvc_packed_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans, 1>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA, Ai, Aj,
                           dx_array_i, dx+(i*stridex), incx, stridex, xi,
                    beta,  dy_array_i, dy+(i*stridey), incy, stridey, yi);
        }
    }
}

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_upper_packed_template_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    T beta, T**       dy_array, T       * dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y );

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(n, TILE_SIZE), 1, ibatch );

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        if (trans == MagmaConjTrans) {
            gemvc_packed_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans, 0>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA, Ai, Aj,
                           dx_array_i, dx+(i*stridex), incx, stridex, xi,
                    beta,  dy_array_i, dy+(i*stridey), incy, stridey, yi);
        }
        else if (trans == MagmaTrans) {
            gemvc_packed_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans, 0>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA, Ai, Aj,
                           dx_array_i, dx+(i*stridex), incx, stridex, xi,
                    beta,  dy_array_i, dy+(i*stridey), incy, stridey, yi);
        }
    }
}

#endif
