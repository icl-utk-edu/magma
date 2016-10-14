/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#ifndef HEMM_TEMPLATE_KERNEL_BATCHED_CUH
#define HEMM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "hemm_template_device.cuh"
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_batched_ll_kernel(
    int M, int N, 
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    
    hemm_template_device_ll
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, 
      Aarray[batchid] + LDA *  coffA + roffA, LDA, 
      Barray[batchid] + LDB *  coffB + roffB, LDB, 
      Carray[batchid] + LDC *  coffC + roffC, LDC, 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_batched_lu_kernel(
    int M, int N, 
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    
    hemm_template_device_lu
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, 
      Aarray[batchid] + LDA *  coffA + roffA, LDA, 
      Barray[batchid] + LDB *  coffB + roffB, LDB, 
      Carray[batchid] + LDC *  coffC + roffC, LDC, 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_batched_rl_kernel(
    int M, int N, 
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    
    hemm_template_device_rl
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, 
      Aarray[batchid] + LDA *  coffA + roffA, LDA, 
      Barray[batchid] + LDB *  coffB + roffB, LDB, 
      Carray[batchid] + LDC *  coffC + roffC, LDC, 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_batched_ru_kernel(
    int M, int N, 
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    
    hemm_template_device_ru
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, 
      Aarray[batchid] + LDA *  coffA + roffA, LDA, 
      Barray[batchid] + LDB *  coffB + roffB, LDB, 
      Carray[batchid] + LDC *  coffC + roffC, LDC, 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
void hemm_template_batched(
    magma_side_t side, magma_uplo_t uplo, 
    magma_int_t m, magma_int_t n, 
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 threads(DIM, DIM, 1);
    dim3 grid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    if( side == MagmaLeft ){
        if(uplo == MagmaLower){
            hemm_template_batched_ll_kernel <T, DIM, BLK_M, BLK_N, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, 
              dA_array, ldda, 
              dB_array, lddb, 
              dC_array, lddc, 
              alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC );
        }else{
            hemm_template_batched_lu_kernel <T, DIM, BLK_M, BLK_N, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, 
              dA_array, ldda, 
              dB_array, lddb, 
              dC_array, lddc, 
              alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC );
        }
    }else{
        if(uplo == MagmaLower){
            hemm_template_batched_rl_kernel <T, DIM, BLK_M, BLK_N, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, 
              dA_array, ldda, 
              dB_array, lddb, 
              dC_array, lddc, 
              alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC );
        }else{
            hemm_template_batched_ru_kernel <T, DIM, BLK_M, BLK_N, CONJA>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, 
              dA_array, ldda, 
              dB_array, lddb, 
              dC_array, lddc, 
              alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC );
        }
    }
}
#endif //HEMM_TEMPLATE_KERNEL_BATCHED_CUH
