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
#ifndef GEMM_TEMPLATE_KERNEL_VBATCHED_CUH
#define GEMM_TEMPLATE_KERNEL_VBATCHED_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "gemm_template_device.dp.hpp"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_vbatched_nn_kernel(
    magma_int_t* M, magma_int_t* N, magma_int_t* K,
    T const * const * Aarray, int Ai, int Aj, magma_int_t* LDA,
    T const * const * Barray, int Bi, int Bj, magma_int_t* LDB,
    T               **Carray, int Ci, int Cj, magma_int_t* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K, sycl::nd_item<3> item_ct1,
    uint8_t*dpct_local)
{
    auto sdata_nn = (T **)dpct_local;

    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < Ai || my_K < Aj ) return;
    if( my_K < Bi || my_N < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Ai, Ci );
    my_N -= max( Bj, Cj );
    my_K -= max( Aj, Bi );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K <= 0) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_M, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = (T*)sdata_nn;        // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_nn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_M, my_N, my_K, Aarray[batchid] + (int)LDA[batchid] * Aj + Ai,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * Bj + Bi,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * Cj + Ci,
        (int)LDC[batchid], alpha, beta, sA, slda, sB, sldb, NULL, 0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_vbatched_nt_kernel(
    magma_int_t* M, magma_int_t* N, magma_int_t* K,
    T const * const * Aarray, int Ai, int Aj, magma_int_t* LDA,
    T const * const * Barray, int Bi, int Bj, magma_int_t* LDB,
    T              ** Carray, int Ci, int Cj, magma_int_t* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K, sycl::nd_item<3> item_ct1,
    uint8_t*dpct_local)
{
    auto sdata_nt = (T **)dpct_local;

    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < Ai || my_K < Aj ) return;
    if( my_N < Bi || my_K < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Ai, Ci );
    my_N -= max( Bi, Cj );
    my_K -= max( Aj, Bj );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K <= 0) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= (my_M + BLK_M - 1) / BLK_M) return;
    if (item_ct1.get_group(1) >= (my_N + BLK_N - 1) / BLK_N) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = (T*)sdata_nt;        // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_nt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_M, my_N, my_K, Aarray[batchid] + (int)LDA[batchid] * Aj + Ai,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * Bj + Bi,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * Cj + Ci,
        (int)LDC[batchid], alpha, beta, sA, slda, sB, sldb, NULL, 0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_vbatched_tn_kernel(
    magma_int_t* M, magma_int_t* N, magma_int_t* K,
    T const * const * Aarray, int Ai, int Aj, magma_int_t* LDA,
    T const * const * Barray, int Bi, int Bj, magma_int_t* LDB,
    T              ** Carray, int Ci, int Cj, magma_int_t* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K, sycl::nd_item<3> item_ct1,
    uint8_t*dpct_local)
{
    auto sdata_tn = (T **)dpct_local;

    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_K < Ai || my_M < Aj ) return;
    if( my_K < Bi || my_N < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Aj, Ci );
    my_N -= max( Bj, Cj );
    my_K -= max( Ai, Bi );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K <= 0) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= (my_M + BLK_M - 1) / BLK_M) return;
    if (item_ct1.get_group(1) >= (my_N + BLK_N - 1) / BLK_N) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = (T*)sdata_tn;        // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_tn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_M, my_N, my_K, Aarray[batchid] + (int)LDA[batchid] * Aj + Ai,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * Bj + Bi,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * Cj + Ci,
        (int)LDC[batchid], alpha, beta, sA, slda, sB, sldb, NULL, 0, item_ct1);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void gemm_template_vbatched_tt_kernel(
    magma_int_t* M, magma_int_t* N, magma_int_t* K,
    T const * const * Aarray, int Ai, int Aj, magma_int_t* LDA,
    T const * const * Barray, int Bi, int Bj, magma_int_t* LDB,
    T              ** Carray, int Ci, int Cj, magma_int_t* LDC,
    T alpha, T beta,
    int max_M, int max_N, int max_K, sycl::nd_item<3> item_ct1,
    uint8_t*dpct_local)
{
    auto sdata_tt = (T **)dpct_local;

    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    int my_K = (int)K[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_K < Ai || my_M < Aj ) return;
    if( my_N < Bi || my_K < Bj ) return;
    if( my_M < Ci || my_N < Cj ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( Aj, Ci );
    my_N -= max( Bi, Cj );
    my_K -= max( Ai, Bj );

    my_M = min( my_M, max_M );
    my_N = min( my_N, max_N );
    my_K = min( my_K, max_K );

    if(my_M <= 0 || my_N <= 0 || my_K <= 0) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= (my_M + BLK_M - 1) / BLK_M) return;
    if (item_ct1.get_group(1) >= (my_N + BLK_N - 1) / BLK_N) return;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = (T*)sdata_tt;        // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_tt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_M, my_N, my_K, Aarray[batchid] + (int)LDA[batchid] * Aj + Ai,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * Bj + Bi,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * Cj + Ci,
        (int)LDC[batchid], alpha, beta, sA, slda, sB, sldb, NULL, 0, item_ct1);
}


/******************************************************************************/
// kernel wrappers
// NN
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_nn(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    T alpha, T const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    T beta,  T              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_m, BLK_M));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        gemm_template_vbatched_nn_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            m + i, n + i, k + i, dA_array + i, Ai, Aj, ldda + i,
                            dB_array + i, Bi, Bj, lddb + i, dC_array + i, Ci,
                            Cj, lddc + i, alpha, beta, max_m, max_n, max_k,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}


/******************************************************************************/
// NT, NC
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_nt(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    T alpha, T const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    T beta,  T              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_m, BLK_M));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        gemm_template_vbatched_nt_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            m + i, n + i, k + i, dA_array + i, Ai, Aj, ldda + i,
                            dB_array + i, Bi, Bj, lddb + i, dC_array + i, Ci,
                            Cj, lddc + i, alpha, beta, max_m, max_n, max_k,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}


/******************************************************************************/
// TN, CN
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_tn(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    T alpha, T const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    T beta,  T              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_m, BLK_M));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        gemm_template_vbatched_tn_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            m + i, n + i, k + i, dA_array + i, Ai, Aj, ldda + i,
                            dB_array + i, Bi, Bj, lddb + i, dC_array + i, Ci,
                            Cj, lddc + i, alpha, beta, max_m, max_n, max_k,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }

}


/******************************************************************************/
// TT, TC, CT, CC
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_vbatched_tt(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    T alpha, T const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    T beta,  T              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_m, BLK_M));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        gemm_template_vbatched_tt_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            m + i, n + i, k + i, dA_array + i, Ai, Aj, ldda + i,
                            dB_array + i, Bi, Bj, lddb + i, dC_array + i, Ci,
                            Cj, lddc + i, alpha, beta, max_m, max_n, max_k,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}

#endif //GEMM_TEMPLATE_KERNEL_VBATCHED_CUH
