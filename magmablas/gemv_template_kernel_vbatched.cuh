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

#include "gemm_template_device_defs.cuh" // use make_FloatingPoint
#include "gemv_template_device.cuh"

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
__global__ void
gemvn_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array, magma_int_t* incx,
    T beta, T**  y_array, magma_int_t* incy)
{
    int batchid = blockIdx.z;

    int my_m = (int)m[batchid];
    if( blockIdx.x >= magma_ceildiv(my_m, TILE_SIZE) ) return;

    gemvn_template_device<T, DIM_X, DIM_Y, TILE_SIZE>
        ( my_m, (int)n[batchid], alpha, A_array[batchid], (int)lda[batchid], x_array[batchid], (int)incx[batchid], beta, y_array[batchid], (int)incy[batchid]);
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
    dim3 threads ( DIM_X, DIM_Y);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(max_m, TILE_SIZE), 1, ibatch );

        gemvn_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( m+i, n+i, alpha, dA_array+i, ldda+i, dx_array+i, incx+i, beta, dy_array+i, incy+i );
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
__global__ void
gemvc_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array,  magma_int_t* incx,
    T beta, T**  y_array, magma_int_t* incy)
{
    int batchid = blockIdx.z;

    int my_n = (int)n[batchid];
    if( blockIdx.x >= magma_ceildiv(my_n, TILE_SIZE) ) return;

    gemvc_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans>
        ( (int)m[batchid], (int)n[batchid], alpha, A_array[batchid], (int)lda[batchid], x_array[batchid], (int)incx[batchid], beta, y_array[batchid], (int)incy[batchid]);
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
    dim3 threads ( DIM_X, DIM_Y );

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid    ( magma_ceildiv(max_n, TILE_SIZE), 1, ibatch );

        if (trans == MagmaConjTrans) {
            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, alpha, dA_array+i, ldda+i, dx_array+i, incx+i, beta, dy_array+i, incy+i );
        }
        else if (trans == MagmaTrans) {
            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, alpha, dA_array+i, ldda+i, dx_array+i, incx+i, beta, dy_array+i, incy+i );
        }
    }
}

#endif  // GEMV_TEMPLATE_KERNEL_VBATCHED_CUH
