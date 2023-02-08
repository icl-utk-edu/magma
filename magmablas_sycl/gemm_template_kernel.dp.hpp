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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "gemm_template_device.dp.hpp"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_nn_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1, uint8_t*dpct_local)
{
    auto sdata_nn = (T **)dpct_local;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = (T*)sdata_nn;        // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_prefetch_nn<
        T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB,
        (BLK_M / DIM_X), (BLK_N / DIM_Y), CONJA, CONJB>(
        M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL,
        0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_nt_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1, uint8_t*dpct_local)
{
    auto sdata_nt = (T **)dpct_local;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_nt;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_prefetch_nt<
        T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB,
        (BLK_M / DIM_X), (BLK_N / DIM_Y), CONJA, CONJB>(
        M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL,
        0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_tn_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1, uint8_t*dpct_local)
{
    auto sdata_tn = (T **)dpct_local;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_tn;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_prefetch_tn<
        T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB,
        (BLK_M / DIM_X), (BLK_N / DIM_Y), CONJA, CONJB>(
        M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL,
        0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_tt_kernel(
    int M, int N, int K,
    T const * A, int LDA,
    T const * B, int LDB,
    T*        C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1, uint8_t*dpct_local)
{
    auto sdata_tt = (T **)dpct_local;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_tt;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_prefetch_tt<
        T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB,
        (BLK_M / DIM_X), (BLK_N / DIM_Y), CONJA, CONJB>(
        M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL,
        0, item_ct1);
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
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N), magma_ceildiv(m, BLK_M));
    /*
    DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) {
                gemm_template_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
                                        DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA,
                                        CONJB>(
                    m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
                    item_ct1, dpct_local_acc_ct1.get_pointer());
            });
    });
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
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N), magma_ceildiv(m, BLK_M));
    /*
    DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) {
                gemm_template_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
                                        DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA,
                                        CONJB>(
                    m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
                    item_ct1, dpct_local_acc_ct1.get_pointer());
            });
    });
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
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N), magma_ceildiv(m, BLK_M));
    /*
    DPCT1049:26: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) {
                gemm_template_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
                                        DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA,
                                        CONJB>(
                    m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
                    item_ct1, dpct_local_acc_ct1.get_pointer());
            });
    });
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
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    sycl::range<3> dimGrid(1, magma_ceildiv(n, BLK_N), magma_ceildiv(m, BLK_M));
    /*
    DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) {
                gemm_template_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
                                        DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA,
                                        CONJB>(
                    m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
                    item_ct1, dpct_local_acc_ct1.get_pointer());
            });
    });
}

#endif //GEMM_TEMPLATE_KERNEL_CUH
