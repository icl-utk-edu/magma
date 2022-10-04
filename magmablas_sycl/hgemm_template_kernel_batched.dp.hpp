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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "hgemm_template_device.dp.hpp"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
static void hgemm_template_batched_nn_kernel(
    int M, int N, int K, sycl::half const *const *Aarray, int LDA,
    sycl::half const *const *Barray, int LDB, T **Carray, int LDC, T alpha,
    T beta, int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(0);
    auto sdata = (float *)dpct_local;

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
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
static void hgemm_template_batched_nt_kernel(
    int M, int N, int K, sycl::half const *const *Aarray, int LDA,
    sycl::half const *const *Barray, int LDB, T **Carray, int LDC, T alpha,
    T beta, int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(0);
    auto sdata = (float *)dpct_local;

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
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
static void hgemm_template_batched_tn_kernel(
    int M, int N, int K, sycl::half const *const *Aarray, int LDA,
    sycl::half const *const *Barray, int LDB, T **Carray, int LDC, T alpha,
    T beta, int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(0);
    auto sdata = (float *)dpct_local;

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
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
static void hgemm_template_batched_tt_kernel(
    int M, int N, int K, sycl::half const *const *Aarray, int LDA,
    sycl::half const *const *Barray, int LDB, T **Carray, int LDC, T alpha,
    T beta, int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(0);
    auto sdata = (float *)dpct_local;

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
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
void hgemm_template_batched_nn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    sycl::half const *const *dA_array, magma_int_t ldda,
    sycl::half const *const *dB_array, magma_int_t lddb, T **dC_array,
    magma_int_t lddc, T alpha, T beta, magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        /*
        DPCT1007:177: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cudaFuncSetAttribute(
            hgemm_template_batched_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N,
                                             BLK_K, TC_M, TC_N, TC_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    sycl::range<3> dimBlock(1, 1, DIM_X * DIM_Y);
    const int maxbatch = queue->get_maxBatch();
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        sycl::range<3> dimGrid(batch, magma_ceildiv(n, BLK_N),
                               magma_ceildiv(m, BLK_M));
        /*
        DPCT1049:176: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        hgemm_template_batched_nn_kernel<T, DIM_X, DIM_Y, BLK_M,
                                                         BLK_N, BLK_K, TC_M,
                                                         TC_N, TC_K>(
                            m, n, k, dA_array + s, ldda, dB_array + s, lddb,
                            dC_array + s, lddc, alpha, beta, roffA, coffA,
                            roffB, coffB, roffC, coffC, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}

/******************************************************************************/
// kernel wrappers
// NT
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
void hgemm_template_batched_nt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    sycl::half const *const *dA_array, magma_int_t ldda,
    sycl::half const *const *dB_array, magma_int_t lddb, T **dC_array,
    magma_int_t lddc, T alpha, T beta, magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        /*
        DPCT1007:179: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cudaFuncSetAttribute(
            hgemm_template_batched_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N,
                                             BLK_K, TC_M, TC_N, TC_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    sycl::range<3> dimBlock(1, 1, DIM_X * DIM_Y);
    const int maxbatch = queue->get_maxBatch();
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        sycl::range<3> dimGrid(batch, magma_ceildiv(n, BLK_N),
                               magma_ceildiv(m, BLK_M));
        /*
        DPCT1049:178: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        hgemm_template_batched_nt_kernel<T, DIM_X, DIM_Y, BLK_M,
                                                         BLK_N, BLK_K, TC_M,
                                                         TC_N, TC_K>(
                            m, n, k, dA_array + s, ldda, dB_array + s, lddb,
                            dC_array + s, lddc, alpha, beta, roffA, coffA,
                            roffB, coffB, roffC, coffC, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}

/******************************************************************************/
// kernel wrappers
// TN
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
void hgemm_template_batched_tn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    sycl::half const *const *dA_array, magma_int_t ldda,
    sycl::half const *const *dB_array, magma_int_t lddb, T **dC_array,
    magma_int_t lddc, T alpha, T beta, magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        /*
        DPCT1007:181: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cudaFuncSetAttribute(
            hgemm_template_batched_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N,
                                             BLK_K, TC_M, TC_N, TC_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    sycl::range<3> dimBlock(1, 1, DIM_X * DIM_Y);
    const int maxbatch = queue->get_maxBatch();
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        sycl::range<3> dimGrid(batch, magma_ceildiv(n, BLK_N),
                               magma_ceildiv(m, BLK_M));
        /*
        DPCT1049:180: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        hgemm_template_batched_tn_kernel<T, DIM_X, DIM_Y, BLK_M,
                                                         BLK_N, BLK_K, TC_M,
                                                         TC_N, TC_K>(
                            m, n, k, dA_array + s, ldda, dB_array + s, lddb,
                            dC_array + s, lddc, alpha, beta, roffA, coffA,
                            roffB, coffB, roffC, coffC, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}

/******************************************************************************/
// kernel wrappers
// TT
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M,
          const int BLK_N, const int BLK_K, const int TC_M, const int TC_N,
          const int TC_K>
void hgemm_template_batched_tt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    sycl::half const *const *dA_array, magma_int_t ldda,
    sycl::half const *const *dB_array, magma_int_t lddb, T **dC_array,
    magma_int_t lddc, T alpha, T beta, magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t shmem = 0;
    shmem += BLK_M * BLK_N * sizeof(T);    // sC
    shmem += BLK_M * BLK_K * sizeof(T);    // sA
    shmem += BLK_K * BLK_N * sizeof(T);    // sB

    #if CUDA_VERSION >= 9000
    if(shmem > 49152) {
        /*
        DPCT1007:183: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cudaFuncSetAttribute(
            hgemm_template_batched_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N,
                                             BLK_K, TC_M, TC_N, TC_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    sycl::range<3> dimBlock(1, 1, DIM_X * DIM_Y);
    const int maxbatch = queue->get_maxBatch();
    for(int s = 0; s < batchCount; s+=maxbatch){
        int batch = min(maxbatch, batchCount-s);
        sycl::range<3> dimGrid(batch, magma_ceildiv(n, BLK_N),
                               magma_ceildiv(m, BLK_M));
        /*
        DPCT1049:182: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        hgemm_template_batched_tt_kernel<T, DIM_X, DIM_Y, BLK_M,
                                                         BLK_N, BLK_K, TC_M,
                                                         TC_N, TC_K>(
                            m, n, k, dA_array + s, ldda, dB_array + s, lddb,
                            dC_array + s, lddc, alpha, beta, roffA, coffA,
                            roffB, coffB, roffC, coffC, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}

#endif //HGEMM_TEMPLATE_KERNEL_BATCHED_CUH
