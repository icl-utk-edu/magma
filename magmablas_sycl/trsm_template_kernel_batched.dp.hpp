/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRSM_TEMPLATE_KERNEL_BATCHED_CUH
#define TRSM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "trsm_template_device.dp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static 

void trsm_template_batched_lNL_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray,  int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    const int batchid = item_ct1.get_group(0);

    trsm_template_device_lNL<T, NB, NRHS>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static 

void trsm_template_batched_lNU_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray,  int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_lNU<T, NB, NRHS>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static 

void trsm_template_batched_lTL_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_lTL<T, NB, NRHS, CONJA>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static 

void trsm_template_batched_lTU_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_lTU<T, NB, NRHS, CONJA>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static 

void trsm_template_batched_rNL_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_rNL<T, NB, NRHS>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
static 

void trsm_template_batched_rNU_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_rNU<T, NB, NRHS>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static 

void trsm_template_batched_rTL_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_rTL<T, NB, NRHS, CONJA>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
static 

void trsm_template_batched_rTU_kernel(
        magma_diag_t diag, int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB, sycl::nd_item<3> item_ct1,
        T *sA, T *sB)
{
    int batchid = item_ct1.get_group(0);

    trsm_template_device_rTU<T, NB, NRHS, CONJA>(
        diag, m, n, alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
        Barray[batchid] + coffB * lddb + roffB, lddb, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
void trsm_template_batched_lNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, NRHS);
    magma_int_t max_batchCount = queue->get_maxBatch();

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NRHS));

            /*
            DPCT1049:1474: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>((NB + 1) * NRHS), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_lNL_kernel<T, NB, NRHS>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NRHS));

            /*
            DPCT1049:1475: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>((NB + 1) * NRHS), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_lNU_kernel<T, NB, NRHS>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
void trsm_template_batched_lTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, NRHS);
    magma_int_t max_batchCount = queue->get_maxBatch();

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NRHS));

            /*
            DPCT1049:1476: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>((NB + 1) * NRHS), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_lTL_kernel<T, NB, NRHS,
                                                             CONJA>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(n, NRHS));

            /*
            DPCT1049:1477: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>((NB + 1) * NRHS), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_lTU_kernel<T, NB, NRHS,
                                                             CONJA>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS>
void trsm_template_batched_rNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, NRHS);
    magma_int_t max_batchCount = queue->get_maxBatch();

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, NRHS));

            /*
            DPCT1049:1478: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(slda * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>(sldb * NB), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_rNL_kernel<T, NB, NRHS>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, NRHS));

            /*
            DPCT1049:1479: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(slda * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>(sldb * NB), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_rNU_kernel<T, NB, NRHS>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int NRHS, const int CONJA>
void trsm_template_batched_rTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, NRHS);
    magma_int_t max_batchCount = queue->get_maxBatch();

    if(uplo == MagmaLower){
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, NRHS));

            /*
            DPCT1049:1480: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(slda * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>(sldb * NB), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_rTL_kernel<T, NB, NRHS,
                                                             CONJA>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }else{
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, magma_ceildiv(m, NRHS));

            /*
            DPCT1049:1481: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<1>(slda * NB), cgh);
                    sycl::accessor<T, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sB_acc_ct1(sycl::range<1>(sldb * NB), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            trsm_template_batched_rTU_kernel<T, NB, NRHS,
                                                             CONJA>(
                                diag, m, n, alpha, dA_array + i, ldda,
                                dB_array + i, lddb, roffA, coffA, roffB, coffB,
                                item_ct1, sA_acc_ct1.get_pointer(),
                                sB_acc_ct1.get_pointer());
                        });
                });
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// template wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB, const int NRHS>
void trsm_small_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        T alpha, T **dA_array, magma_int_t ldda,
                 T **dB_array, magma_int_t lddb,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t shape = 0;
    if      (side == MagmaLeft  && transA == MagmaNoTrans   ) { shape = 0; } // left  - NoTrans   (lNx)
    else if (side == MagmaLeft  && transA == MagmaTrans     ) { shape = 1; } // left  - Trans     (lTx)
    else if (side == MagmaLeft  && transA == MagmaConjTrans ) { shape = 2; } // left  - ConjTrans (lCx)
    else if (side == MagmaRight && transA == MagmaNoTrans   ) { shape = 3; } // right - NoTrans   (rNx)
    else if (side == MagmaRight && transA == MagmaTrans     ) { shape = 4; } // right - Trans     (rTx)
    else if (side == MagmaRight && transA == MagmaConjTrans ) { shape = 5; } // right - ConjTrans (rCx)

    switch(shape)
    {
        case 0: // lNx
            trsm_template_batched_lNx<T, NB, NRHS>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, roffA, coffA, roffB, coffB, batchCount, queue);
            break;
        case 1: // lTx
            trsm_template_batched_lTx<T, NB, NRHS, 0>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, roffA, coffA, roffB, coffB, batchCount, queue);
            break;
        case 2: // lCx
            trsm_template_batched_lTx<T, NB, NRHS, 1>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, roffA, coffA, roffB, coffB, batchCount, queue);
            break;
        case 3: // rNx
            trsm_template_batched_rNx<T, NB, NRHS>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, roffA, coffA, roffB, coffB, batchCount, queue);
            break;
        case 4: // rTx
            trsm_template_batched_rTx<T, NB, NRHS, 0>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, roffA, coffA, roffB, coffB, batchCount, queue);
            break;
        case 5: // rCx
            trsm_template_batched_rTx<T, NB, NRHS, 1>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, roffA, coffA, roffB, coffB, batchCount, queue);
            break;
        default:; // propose something
    }
}

#endif //TRSM_TEMPLATE_KERNEL_BATCHED_CUH
