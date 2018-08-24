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

#ifndef GEMM_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMM_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.cuh"
#include "gemm_template_device.cuh"


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_nn_kernel(
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
    
    gemm_template_device_nn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_nt_kernel(
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
    
    gemm_template_device_nt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_tn_kernel(
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
    
    gemm_template_device_tn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_tt_kernel(
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
    
    gemm_template_device_tt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K, 
          Aarray[batchid] + LDA *  coffA + roffA, LDA, 
          Barray[batchid] + LDB *  coffB + roffB, LDB, 
          Carray[batchid] + LDC *  coffC + roffC, LDC, 
          alpha, beta );
}


/******************************************************************************/
// kernel wrappers
// NN 
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,  
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_nn(
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
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 dimBlock(DIM_X, DIM_Y);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), this_batchCount );
        gemm_template_batched_nn_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
            <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
            (m, n, k, dA_array+batch_starting_id, ldda, dB_array+batch_starting_id, lddb, dC_array+batch_starting_id, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}


/******************************************************************************/
// NT, NC 
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_nt(
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
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 dimBlock(DIM_X, DIM_Y);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), this_batchCount );
        gemm_template_batched_nt_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
            <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
            (m, n, k, dA_array+batch_starting_id, ldda, dB_array+batch_starting_id, lddb, dC_array+batch_starting_id, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}


/******************************************************************************/
// TN, CN 
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_tn(
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
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 dimBlock(DIM_X, DIM_Y);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), this_batchCount );
        gemm_template_batched_tn_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
            <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
            (m, n, k, dA_array+batch_starting_id, ldda, dB_array+batch_starting_id, lddb, dC_array+batch_starting_id, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}


/******************************************************************************/
// TT, TC, CT, CC
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_tt(
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
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 dimBlock(DIM_X, DIM_Y);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), this_batchCount );

        gemm_template_batched_tt_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
            <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
            (m, n, k, dA_array+batch_starting_id, ldda, dB_array+batch_starting_id, lddb, dC_array+batch_starting_id, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}

#endif //GEMM_TEMPLATE_KERNEL_BATCHED_CUH
