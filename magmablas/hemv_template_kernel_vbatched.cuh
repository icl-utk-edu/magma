/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
*/

#ifndef HEMV_TEMPLATE_KERNEL_VBATCHED_CUH
#define HEMV_TEMPLATE_KERNEL_VBATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "atomics.cuh"
#include "hemv_template_device.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static __global__
void hemv_diag_template_vbatched_kernel(
        magma_uplo_t uplo, magma_int_t* n, 
        T alpha, T** Aarray, magma_int_t* ldda, 
                 T** Xarray, magma_int_t* incx,
        T beta,  T** Yarray, magma_int_t* incy, 
        int max_N, 
        int offA, int offX, int offY, 
        int spec_N)
{
    const int batchid = blockIdx.z;
    int my_N = (int)n[batchid];
    // check if the offset produces an out-of-bound pointer
    if( my_N < offA) return;
    // compute the maximum allowed n
    my_N -= offA;
    // check if the user forces n
    my_N = ( spec_N <= 0 ) ? my_N : min( my_N, spec_N );
    
    if( my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Xarray[batchid] == NULL || Yarray[batchid] == NULL ) return;
    if(blockIdx.x >= magma_ceildiv(my_N, NB)) return;
    hemv_diag_device<T, NB, TY>( uplo, my_N, 
                                 alpha, Aarray[batchid] + offA * (int)ldda[batchid] + offA, (int)ldda[batchid], 
                                        Xarray[batchid] + offX * (int)incx[batchid], (int)incx[batchid], 
                                 beta , Yarray[batchid] + offY * (int)incy[batchid], (int)incy[batchid] );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static __global__
void hemv_lower_template_vbatched_kernel(
        magma_int_t* n, T alpha, 
        T** Aarray, magma_int_t* ldda, 
        T** Xarray, magma_int_t* incx,
        T** Yarray, magma_int_t* incy, 
        int max_N, 
        int offA, int offX, int offY, 
        int spec_N)
{
    const int batchid = blockIdx.z;
    int my_N = (int)n[batchid];
    // check if the offset produces an out-of-bound pointer
    if( my_N < offA) return;
    // compute the maximum allowed n
    my_N -= offA;
    // check if the user forces n
    my_N = ( spec_N <= 0 ) ? my_N : min( my_N, spec_N );
    
    if( my_N <= NB ) return;    // sizes <= NB are handled by the diag kernel
    if( Aarray[batchid] == NULL || Xarray[batchid] == NULL || Yarray[batchid] == NULL ) return;
    if(blockIdx.x >= magma_ceildiv(my_N, NB)) return;
    hemv_lower_device<T, NB, TY>( my_N, alpha, 
                                  Aarray[batchid] + offA * (int)ldda[batchid] + offA, (int)ldda[batchid], 
                                  Xarray[batchid] + offX * (int)incx[batchid], (int)incx[batchid], 
                                  Yarray[batchid] + offY * (int)incy[batchid], (int)incy[batchid] );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static __global__
void hemv_upper_template_vbatched_kernel(
        magma_int_t* n, T alpha, 
        T** Aarray, magma_int_t* ldda, 
        T** Xarray, magma_int_t* incx,
        T** Yarray, magma_int_t* incy, 
        int max_N, 
        int offA, int offX, int offY, 
        int spec_N)
{
    const int batchid = blockIdx.z;
    int my_N = (int)n[batchid];
    // check if the offset produces an out-of-bound pointer
    if( my_N < offA) return;
    // compute the maximum allowed n
    my_N -= offA;
    // check if the user forces n
    my_N = ( spec_N <= 0 ) ? my_N : min( my_N, spec_N );
    
    if( my_N <= NB ) return;    // sizes <= NB are handled by the diag kernel
    if( Aarray[batchid] == NULL || Xarray[batchid] == NULL || Yarray[batchid] == NULL ) return;
    if(blockIdx.x >= magma_ceildiv(my_N, NB)) return;
    hemv_upper_device<T, NB, TY>( my_N, alpha, 
                                  Aarray[batchid] + offA * (int)ldda[batchid] + offA, (int)ldda[batchid], 
                                  Xarray[batchid] + offX * (int)incx[batchid], (int)incx[batchid], 
                                  Yarray[batchid] + offY * (int)incy[batchid], (int)incy[batchid] );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// diag
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_diag_template_vbatched(
    magma_uplo_t uplo, magma_int_t* n, 
    T alpha, T** dA_array, magma_int_t* ldda,
             T** dX_array, magma_int_t* incx,
    T beta,  T** dY_array, magma_int_t* incy,  
    magma_int_t max_n, 
    magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t spec_n, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 threads(NB, TY, 1);
    dim3 grid( magma_ceildiv( max_n, NB ), 1, batchCount );
    hemv_diag_template_vbatched_kernel<T, NB, TY>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( uplo, n, 
          alpha, dA_array, ldda, 
                 dX_array, incx, 
          beta,  dY_array, incy, 
          max_n, offA, offX, offY, spec_n);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lower
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_lower_template_vbatched(
    magma_int_t* n, T alpha, 
    T** dA_array, magma_int_t* ldda,
    T** dX_array, magma_int_t* incx,
    T** dY_array, magma_int_t* incy, 
    magma_int_t max_n,  
    magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t spec_n, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 threads(NB, TY, 1);
    dim3 grid( magma_ceildiv( max_n, NB ), 1, batchCount );
    hemv_lower_template_vbatched_kernel<T, NB, TY>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( n, alpha, 
          dA_array, ldda, 
          dX_array, incx, 
          dY_array, incy, 
          max_n, offA, offX, offY, spec_n);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// upper
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_upper_template_vbatched(
    magma_int_t* n, T alpha, 
    T** dA_array, magma_int_t* ldda,
    T** dX_array, magma_int_t* incx,
    T** dY_array, magma_int_t* incy, 
    magma_int_t max_n,  
    magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t spec_n, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 threads(NB, TY, 1);
    dim3 grid( magma_ceildiv( max_n, NB ), 1, batchCount );
    hemv_upper_template_vbatched_kernel<T, NB, TY>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( n, alpha, 
          dA_array, ldda, 
          dX_array, incx, 
          dY_array, incy, 
          max_n, offA, offX, offY, spec_n);
}
#endif //HEMV_TEMPLATE_KERNEL_VBATCHED_CUH
