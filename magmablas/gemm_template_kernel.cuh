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

#ifndef GEMM_TEMPLATE_KERNEL_CUH
#define GEMM_TEMPLATE_KERNEL_CUH

#include "gemm_template_device_defs.cuh"
#include "gemm_template_device.cuh"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_nn_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta )
{
    extern __shared__ T* sdata_nn[];

    const int slda = SLDA(BLK_M);
    const int sldb = SLDB(BLK_K);
    T* sA = (T*)sdata_nn;        // sA is slda x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is sldb x (BLK_N)

    gemm_template_device_nn
    <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_nt_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta )
{
    extern __shared__ T* sdata_nt[];

    const int slda = SLDA(BLK_M);
    const int sldb = SLDB(BLK_K);
    T* sA = (T*)sdata_nt;      // sA is slda x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is sldb x (BLK_N)

    gemm_template_device_nt
    <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_tn_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta )
{
    extern __shared__ T* sdata_tn[];

    const int slda = SLDA(BLK_M);
    const int sldb = SLDB(BLK_K);
    T* sA = (T*)sdata_tn;      // sA is slda x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is sldb x (BLK_N)

    gemm_template_device_tn
    <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_tt_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta )
{
    extern __shared__ T* sdata_tt[];

    const int slda = SLDA(BLK_M);
    const int sldb = SLDB(BLK_K);
    T* sA = (T*)sdata_tt;      // sA is slda x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is sldb x (BLK_N)

    gemm_template_device_tt
    <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
// kernel wrappers
// NN
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_nn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * dA, magma_int_t ldda,
    T const * dB, magma_int_t lddb,
    T*        dC, magma_int_t lddc,
    T alpha, T beta, magma_queue_t queue)
{
    size_t shmem = 0;
    shmem += SLDA(BLK_M) * BLK_K * sizeof(T);  // sA
    shmem += SLDB(BLK_K) * BLK_N * sizeof(T);  // sB

    #if CUDA_VERSION >= 9000
    // always opt-in for shared memory
    cudaFuncSetAttribute(
            gemm_template_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    #endif

    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), 1 );
    gemm_template_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
    <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
    (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta);
}


/******************************************************************************/
// NT, NC
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_nt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * dA, magma_int_t ldda,
    T const * dB, magma_int_t lddb,
    T*        dC, magma_int_t lddc,
    T alpha, T beta, magma_queue_t queue)
{
    size_t shmem = 0;
    shmem += SLDA(BLK_M) * BLK_K * sizeof(T);  // sA
    shmem += SLDB(BLK_K) * BLK_N * sizeof(T);  // sB

    #if CUDA_VERSION >= 9000
    // always opt-in for shared memory
    cudaFuncSetAttribute(
            gemm_template_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    #endif

    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), 1 );
    gemm_template_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
    <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
    (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta);
}


/******************************************************************************/
// TN, CN
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_tn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * dA, magma_int_t ldda,
    T const * dB, magma_int_t lddb,
    T*        dC, magma_int_t lddc,
    T alpha, T beta, magma_queue_t queue)
{
    size_t shmem = 0;
    shmem += SLDA(BLK_M) * BLK_K * sizeof(T);  // sA
    shmem += SLDB(BLK_K) * BLK_N * sizeof(T);  // sB

    #if CUDA_VERSION >= 9000
    // always opt-in for shared memory
    cudaFuncSetAttribute(
            gemm_template_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    #endif

    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), 1 );
    gemm_template_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
    <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
    (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta);
}


/******************************************************************************/
// TT, TC, CT, CC
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_tt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * dA, magma_int_t ldda,
    T const * dB, magma_int_t lddb,
    T*        dC, magma_int_t lddc,
    T alpha, T beta, magma_queue_t queue)
{
    size_t shmem = 0;
    shmem += SLDA(BLK_M) * BLK_K * sizeof(T);  // sA
    shmem += SLDB(BLK_K) * BLK_N * sizeof(T);  // sB

    #if CUDA_VERSION >= 9000
    // always opt-in for shared memory
    cudaFuncSetAttribute(
            gemm_template_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    #endif

    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), 1 );
    gemm_template_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
    <<< dimGrid, dimBlock, shmem, queue->cuda_stream() >>>
    (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta);
}

#endif //GEMM_TEMPLATE_KERNEL_CUH
