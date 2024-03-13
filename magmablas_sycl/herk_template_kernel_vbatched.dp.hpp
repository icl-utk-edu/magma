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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "gemm_template_device.dp.hpp"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void herk_template_vbatched_nt_kernel(
    magma_uplo_t uplo, magma_int_t* N, magma_int_t* K,
    T alpha,
    T const * const * Aarray, magma_int_t* LDA,
    T const * const * Barray, magma_int_t* LDB,
    T beta, T**       Carray, magma_int_t* LDC, sycl::nd_item<3> item_ct1,
    uint8_t*dpct_local)
{
    auto sdata_nt = (T **)dpct_local;
    const int batchid = item_ct1.get_group(0);
    const int my_N = (int)N[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ((uplo == MagmaLower) &&
        (item_ct1.get_group(1) * BLK_N > (item_ct1.get_group(2) + 1) * BLK_M))
        return;

    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ((uplo == MagmaUpper) &&
        (item_ct1.get_group(2) * BLK_M > (item_ct1.get_group(1) + 1) * BLK_N))
        return;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_nt;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)
    gemm_template_device_nt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_N, my_N, (int)K[batchid], Aarray[batchid], (int)LDA[batchid],
        Barray[batchid], (int)LDB[batchid], Carray[batchid], (int)LDC[batchid],
        alpha, beta, sA, slda, sB, sldb, NULL, 0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void herk_template_vbatched_tn_kernel(
    magma_uplo_t uplo, magma_int_t* N, magma_int_t* K,
    T alpha, T const * const * Aarray, magma_int_t* LDA,
    T const * const * Barray, magma_int_t* LDB,
    T beta, T**       Carray, magma_int_t* LDC , sycl::nd_item<3> item_ct1,
    uint8_t*dpct_local)
{
    auto sdata_tn = (T **)dpct_local;
    const int batchid = item_ct1.get_group(0);
    const int my_N = (int)N[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ((uplo == MagmaLower) &&
        (item_ct1.get_group(1) * BLK_N > (item_ct1.get_group(2) + 1) * BLK_M))
        return;

    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ((uplo == MagmaUpper) &&
        (item_ct1.get_group(2) * BLK_M > (item_ct1.get_group(1) + 1) * BLK_N))
        return;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_tn;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)
    gemm_template_device_tn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_N, my_N, (int)K[batchid], Aarray[batchid], (int)LDA[batchid],
        Barray[batchid], (int)LDB[batchid], Carray[batchid], (int)LDC[batchid],
        alpha, beta, sA, slda, sB, sldb, NULL, 0, item_ct1);
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
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_n, BLK_M));
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        herk_template_vbatched_nt_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            uplo, n + i, k + i, alpha, dA_array + i, ldda + i,
                            dB_array + i, lddb + i, beta, dC_array + i,
                            lddc + i, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
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
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_n, BLK_M));
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        herk_template_vbatched_tn_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            uplo, n + i, k + i, alpha, dA_array + i, ldda + i,
                            dB_array + i, lddb + i, beta, dC_array + i,
                            lddc + i, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}

#endif //HERK_TEMPLATE_KERNEL_VBATCHED_CUH
