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
#include <CL/sycl.hpp>
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

        /*
        DPCT1049:1438: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     trmm_template_batched_lNx_kernel<T, NB>(
                                         uplo, diag, m, n, alpha, dA_array + i,
                                         ldda, dB_array + i, lddb, roffA, coffA,
                                         roffB, coffB, item_ct1,
                                         sA_acc_ct1.get_pointer(),
                                         sB_acc_ct1.get_pointer());
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

        /*
        DPCT1049:1439: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        trmm_template_batched_lTx_kernel<T, NB, CONJA>(
                            uplo, diag, m, n, alpha, dA_array + i, ldda,
                            dB_array + i, lddb, roffA, coffA, roffB, coffB,
                            item_ct1, sA_acc_ct1.get_pointer(),
                            sB_acc_ct1.get_pointer());
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

        /*
        DPCT1049:1440: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     trmm_template_batched_rNx_kernel<T, NB>(
                                         uplo, diag, m, n, alpha, dA_array + i,
                                         ldda, dB_array + i, lddb, roffA, coffA,
                                         roffB, coffB, item_ct1,
                                         sA_acc_ct1.get_pointer(),
                                         sB_acc_ct1.get_pointer());
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
        /*
        DPCT1049:1441: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        trmm_template_batched_rTx_kernel<T, NB, CONJA>(
                            uplo, diag, m, n, alpha, dA_array + i, ldda,
                            dB_array + i, lddb, roffA, coffA, roffB, coffB,
                            item_ct1, sA_acc_ct1.get_pointer(),
                            sB_acc_ct1.get_pointer());
                    });
            });
    }
}
#endif //TRMM_TEMPLATE_KERNEL_BATCHED_CUH
