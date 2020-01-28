/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/
#ifndef HERK_TEMPLATE_KERNEL_VBATCHED_CUH
#define HERK_TEMPLATE_KERNEL_VBATCHED_CUH

#include "gemm_template_device_defs.cuh"
#include "gemm_template_device.cuh"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void herk_template_vbatched_nt_kernel(
    magma_uplo_t uplo, magma_int_t* N, magma_int_t* K, 
    T alpha, 
    T const * const * Aarray, magma_int_t* LDA,
    T const * const * Barray, magma_int_t* LDB,
    T beta, T**       Carray, magma_int_t* LDC)
{
    const int batchid = blockIdx.z; 
    const int my_N = (int)N[batchid];
    if( blockIdx.x >= magma_ceildiv( my_N, BLK_M ) ) return;
    if( blockIdx.y >= magma_ceildiv( my_N, BLK_N ) ) return;
    
    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ( ( uplo == MagmaLower ) && ( blockIdx.y*BLK_N > (blockIdx.x+1)*BLK_M ) )
        return;
    
    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ( ( uplo == MagmaUpper)  && ( blockIdx.x*BLK_M > (blockIdx.y+1)*BLK_N ) )
        return;

    gemm_template_device_nt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( my_N, my_N, (int)K[batchid], Aarray[batchid], (int)LDA[batchid], Barray[batchid], (int)LDB[batchid], Carray[batchid], (int)LDC[batchid], alpha, beta );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void herk_template_vbatched_tn_kernel(
    magma_uplo_t uplo, magma_int_t* N, magma_int_t* K, 
    T alpha, T const * const * Aarray, magma_int_t* LDA,
    T const * const * Barray, magma_int_t* LDB,
    T beta, T**       Carray, magma_int_t* LDC )
{
    const int batchid = blockIdx.z; 
    const int my_N = (int)N[batchid];
    if( blockIdx.x >= magma_ceildiv( my_N, BLK_M ) ) return;
    if( blockIdx.y >= magma_ceildiv( my_N, BLK_N ) ) return;

    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ( ( uplo == MagmaLower ) && ( blockIdx.y*BLK_N > (blockIdx.x+1)*BLK_M ) )
        return;
    
    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ( ( uplo == MagmaUpper)  && ( blockIdx.x*BLK_M > (blockIdx.y+1)*BLK_N ) )
        return;

    gemm_template_device_tn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
    ( my_N, my_N, (int)K[batchid], Aarray[batchid], (int)LDA[batchid], Barray[batchid], (int)LDB[batchid], Carray[batchid], (int)LDC[batchid], alpha, beta );
}


/******************************************************************************/
// kernel wrappers
// NT, NC 
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void herk_template_vbatched_nt(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t* k,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dB_array, magma_int_t* lddb,
    T**       dC_array, magma_int_t* lddc,
    T alpha, T beta,
    magma_int_t batchCount, magma_queue_t queue, 
    magma_int_t max_n)
{
    magma_int_t max_batchCount = 50000;
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( max_n, BLK_M ), magma_ceildiv( max_n, BLK_N ), ibatch );
        herk_template_vbatched_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<<dimGrid, dimBlock, 0, queue->cuda_stream()>>>
        (uplo, n+i, k+i, alpha, dA_array+i, ldda+i, dB_array+i, lddb+i, beta, dC_array+i, lddc+i);
    }
}


/******************************************************************************/
// TN, CN 
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void herk_template_vbatched_tn(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t* k,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dB_array, magma_int_t* lddb,
    T**       dC_array, magma_int_t* lddc,
    T alpha, T beta,
    magma_int_t batchCount, magma_queue_t queue, 
    magma_int_t max_n)
{
    magma_int_t max_batchCount = 50000;
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( max_n, BLK_M ), magma_ceildiv( max_n, BLK_N ), ibatch );
        herk_template_vbatched_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<<dimGrid, dimBlock, 0, queue->cuda_stream()>>>
        (uplo, n+i, k+i, alpha, dA_array+i, ldda+i, dB_array+i, lddb+i, beta, dC_array+i, lddc+i);
    }
}

#endif //HERK_TEMPLATE_KERNEL_VBATCHED_CUH
