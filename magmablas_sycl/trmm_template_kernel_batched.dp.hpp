/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_KERNEL_BATCHED_CUH
#define TRMM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "trmm_template_device.dp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 
void trmm_template_batched_lNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray,  int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trmm_small_template_device_lNx<T, NB>(
        uplo, diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 
void trmm_template_batched_lTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trmm_small_template_device_lTx<T, NB, CONJA>(
        uplo, diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 
void trmm_template_batched_rNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trmm_small_template_device_rNx<T, NB>(
        uplo, diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 
void trmm_template_batched_rTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trmm_small_template_device_rTx<T, NB, CONJA>(
        uplo, diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_batched_lNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::local_accessor<T, 1>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     trmm_template_batched_lNx_kernel<T, NB>(
                                         uplo, diag, m, n, alpha, dA_array + i,
                                         ldda, dB_array + i, lddb, roffA, coffA,
                                         roffB, coffB, item_ct1,
                                         sA_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					 sB_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_batched_lTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::local_accessor<T, 1>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        trmm_template_batched_lTx_kernel<T, NB, CONJA>(
                            uplo, diag, m, n, alpha, dA_array + i, ldda,
                            dB_array + i, lddb, roffA, coffA, roffB, coffB,
                            item_ct1, sA_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
			    sB_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_batched_rNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, NB));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::local_accessor<T, 1>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     trmm_template_batched_rNx_kernel<T, NB>(
                                         uplo, diag, m, n, alpha, dA_array + i,
                                         ldda, dB_array + i, lddb, roffA, coffA,
                                         roffB, coffB, item_ct1,
                                         sA_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
					 sB_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_batched_rTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);

        sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, NB));
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<T, 1>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::local_accessor<T, 1>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        trmm_template_batched_rTx_kernel<T, NB, CONJA>(
                            uplo, diag, m, n, alpha, dA_array + i, ldda,
                            dB_array + i, lddb, roffA, coffA, roffB, coffB,
                            item_ct1, sA_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
			    sB_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
}
#endif //TRMM_TEMPLATE_KERNEL_BATCHED_CUH
