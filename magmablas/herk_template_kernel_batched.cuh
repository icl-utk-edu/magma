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
#ifndef HERK_TEMPLATE_KERNEL_BATCHED_CUH
#define HERK_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.cuh"
#include "gemm_template_device.cuh"


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void herk_template_batched_nt_kernel(
    magma_uplo_t uplo, int N, int K, 
    T alpha, 
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T beta, T**       Carray, int LDC, 
    int Ai, int Aj, 
    int Bi, int Bj, 
    int Ci, int Cj)
{
    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ( ( uplo == MagmaLower ) && ( blockIdx.y*BLK_N > (blockIdx.x+1)*BLK_M ) )
        return;
    
    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ( ( uplo == MagmaUpper)  && ( blockIdx.x*BLK_M > (blockIdx.y+1)*BLK_N ) )
        return;

    int batchid = blockIdx.z;
    
    gemm_template_device_nt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( N, N, K, Aarray[batchid] + Aj * LDA + Ai, LDA, 
                   Barray[batchid] + Bj * LDB + Bi, LDB, 
                   Carray[batchid] + Cj * LDC + Ci, LDC, 
                   alpha, beta );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void herk_template_batched_tn_kernel(
    magma_uplo_t uplo, int N, int K, 
    T alpha, 
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T beta, T**       Carray, int LDC, 
    int Ai, int Aj, 
    int Bi, int Bj, 
    int Ci, int Cj)

{
    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ( ( uplo == MagmaLower ) && ( blockIdx.y*BLK_N > (blockIdx.x+1)*BLK_M ) )
        return;
    
    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ( ( uplo == MagmaUpper)  && ( blockIdx.x*BLK_M > (blockIdx.y+1)*BLK_N ) )
        return;

    int batchid = blockIdx.z;
    
    gemm_template_device_tn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( N, N, K, Aarray[batchid] + Aj * LDA + Ai, LDA, 
                   Barray[batchid] + Bj * LDB + Bi, LDB, 
                   Carray[batchid] + Cj * LDC + Ci, LDC,
                   alpha, beta );
}


/******************************************************************************/
// kernel wrappers
// NT, NC 
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void herk_template_batched_nt(
    magma_uplo_t uplo, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    T const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    T**               dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( n, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    herk_template_batched_nt_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (uplo, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, ai, aj, bi, bj, ci, cj);
}


/******************************************************************************/
// TN, CN 
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void herk_template_batched_tn(
    magma_uplo_t uplo, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    T const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    T**               dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( n, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    herk_template_batched_tn_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (uplo, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, ai, aj, bi, bj, ci, cj);
}

#endif //HERK_TEMPLATE_KERNEL_BATCHED_CUH
