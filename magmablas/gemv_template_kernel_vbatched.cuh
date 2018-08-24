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
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 grid    ( magma_ceildiv(max_m, TILE_SIZE), 1, this_batchCount );                                                
        dim3 threads ( DIM_X, DIM_Y);

        gemvn_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE>
            <<< grid, threads, 0, queue->cuda_stream() >>>                    
            ( m, n, alpha, dA_array+batch_starting_id, ldda, dx_array+batch_starting_id, incx, beta, dy_array+batch_starting_id, incy );
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
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 grid    ( magma_ceildiv(max_n, TILE_SIZE), 1, this_batchCount );                                                
        dim3 threads ( DIM_X, DIM_Y );

        if (trans == MagmaConjTrans)
        {                         
            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans>
                <<< grid, threads, 0, queue->cuda_stream() >>>                    
                ( m, n, alpha, dA_array+batch_starting_id, ldda, dx_array+batch_starting_id, incx, beta, dy_array+batch_starting_id, incy );        
        }
        else if (trans == MagmaTrans)
        {
            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans>
                <<< grid, threads, 0, queue->cuda_stream() >>>                    
                ( m, n, alpha, dA_array+batch_starting_id, ldda, dx_array+batch_starting_id, incx, beta, dy_array+batch_starting_id, incy );       
        }
    }
}

#endif  // GEMV_TEMPLATE_KERNEL_VBATCHED_CUH
