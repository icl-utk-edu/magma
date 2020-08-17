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
#include "gemm_template_device_defs.cuh"
#include "atomics.cuh"
#include "hemv_template_device.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static __global__
void hemv_diag_template_batched_kernel(
        magma_uplo_t uplo, int n,
        T alpha, T** Aarray, int ldda,
                 T** Xarray, int incx,
        T beta,  T** Yarray, int incy,
        int offA, int offX, int offY)
{
    int batchid = blockIdx.z;

    hemv_diag_device<T, NB, TY>( uplo, n,
                                 alpha, Aarray[batchid] + offA * ldda + offA, ldda,
                                        Xarray[batchid] + offX * incx, incx,
                                 beta , Yarray[batchid] + offY * incy, incy );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static __global__
void hemv_lower_template_batched_kernel(
        int n, T alpha,
        T** Aarray, int ldda,
        T** Xarray, int incx,
        T** Yarray, int incy,
        int offA, int offX, int offY)
{
    int batchid = blockIdx.z;

    hemv_lower_device<T, NB, TY>( n, alpha,
                                  Aarray[batchid] + offA * ldda + offA, ldda,
                                  Xarray[batchid] + offX * incx, incx,
                                  Yarray[batchid] + offY * incy, incy );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static __global__
void hemv_upper_template_batched_kernel(
        int n, T alpha,
        T** Aarray, int ldda,
        T** Xarray, int incx,
        T** Yarray, int incy,
        int offA, int offX, int offY)
{
    int batchid = blockIdx.z;

    hemv_upper_device<T, NB, TY>( n, alpha,
                                  Aarray[batchid] + offA * ldda + offA, ldda,
                                  Xarray[batchid] + offX * incx, incx,
                                  Yarray[batchid] + offY * incy, incy );
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
    dim3 threads(NB, TY, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, NB ), 1, ibatch );

        hemv_diag_template_batched_kernel<T, NB, TY>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( uplo, n,
          alpha, dA_array+i, ldda,
                 dX_array+i, incx,
          beta,  dY_array+i, incy,
          offA, offX, offY );
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
    dim3 threads(NB, TY, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, NB ), 1, ibatch );

        hemv_lower_template_batched_kernel<T, NB, TY>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( n, alpha,
          dA_array+i, ldda,
          dX_array+i, incx,
          dY_array+i, incy,
          offA, offX, offY );
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
    dim3 threads(NB, TY, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, NB ), 1, ibatch );

        hemv_upper_template_batched_kernel<T, NB, TY>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( n, alpha,
          dA_array+i, ldda,
          dX_array+i, incx,
          dY_array+i, incy,
          offA, offX, offY);
    }
}
#endif //HEMV_TEMPLATE_KERNEL_BATCHED_CUH
