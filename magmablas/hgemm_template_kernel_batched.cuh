/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
*/

#ifndef HGEMM_TEMPLATE_KERNEL_BATCHED_CUH
#define HGEMM_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.cuh"
#include "hgemm_template_device.cuh"

extern __shared__ float sdata[];
/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __global__
void hgemm_template_batched_nn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    T* sC = (T*)sdata;
    T* sA = sC + BLK_M * BLK_N;
    T* sB = sA + BLK_M * BLK_K;    
    hgemm_template_device_nn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta, 
          sA, sB, sC );
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __global__
void hgemm_template_batched_nt_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    T* sC = (T*)sdata;
    T* sA = sC + BLK_M * BLK_N;
    T* sB = sA + BLK_M * BLK_K;    
    hgemm_template_device_nt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta, 
          sA, sB, sC );
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __global__
void hgemm_template_batched_tn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    T* sC = (T*)sdata;
    T* sA = sC + BLK_M * BLK_N;
    T* sB = sA + BLK_M * BLK_K;    
    hgemm_template_device_tn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta, 
          sA, sB, sC );
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __global__
void hgemm_template_batched_tt_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta, 
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    const int batchid = blockIdx.z;
    T* sC = (T*)sdata;
    T* sA = sC + BLK_M * BLK_N;
    T* sB = sA + BLK_M * BLK_K;    
    hgemm_template_device_tt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta, 
          sA, sB, sC );
}

/******************************************************************************/
// kernel wrappers
// NN 
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
void hgemm_template_batched_nn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta, 
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        cudaFuncSetAttribute( hgemm_template_batched_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>, 
                              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    dim3 dimBlock(DIM_X * DIM_Y, 1);
    const int maxbatch = 50000;
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batch );
        hgemm_template_batched_nn_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
            <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
            (m, n, k, dA_array+s, ldda, dB_array+s, lddb, dC_array+s, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}

/******************************************************************************/
// kernel wrappers
// NT
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
void hgemm_template_batched_nt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta, 
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        cudaFuncSetAttribute( hgemm_template_batched_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>, 
                              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    dim3 dimBlock(DIM_X * DIM_Y, 1);
    const int maxbatch = 50000;
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batch );
        hgemm_template_batched_nt_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
            <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
            (m, n, k, dA_array+s, ldda, dB_array+s, lddb, dC_array+s, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}

/******************************************************************************/
// kernel wrappers
// TN
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
void hgemm_template_batched_tn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta, 
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        cudaFuncSetAttribute( hgemm_template_batched_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>, 
                              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    dim3 dimBlock(DIM_X * DIM_Y, 1);
    const int maxbatch = 50000;
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batch );
        hgemm_template_batched_tn_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
            <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
            (m, n, k, dA_array+s, ldda, dB_array+s, lddb, dC_array+s, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}

/******************************************************************************/
// kernel wrappers
// TT
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
void hgemm_template_batched_tt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta, 
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        cudaFuncSetAttribute( hgemm_template_batched_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>, 
                              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    dim3 dimBlock(DIM_X * DIM_Y, 1);
    const int maxbatch = 50000;
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batch );
        hgemm_template_batched_tt_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K>
            <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
            (m, n, k, dA_array+s, ldda, dB_array+s, lddb, dC_array+s, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}

#endif //HGEMM_TEMPLATE_KERNEL_BATCHED_CUH
