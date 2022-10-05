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

#ifndef HEMM_TEMPLATE_KERNEL_BATCHED_CUH
#define HEMM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "hemm_template_device.dp.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_batched_ll_kernel(
    int M, int N,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC ,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    const int batchid = item_ct1.get_group(0);

    hemm_template_device_ll<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, Aarray[batchid] + LDA * coffA + roffA,
                                   LDA, Barray[batchid] + LDB * coffB + roffB,
                                   LDB, Carray[batchid] + LDC * coffC + roffC,
                                   LDC, alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_batched_lu_kernel(
    int M, int N,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC ,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    const int batchid = item_ct1.get_group(0);

    hemm_template_device_lu<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, Aarray[batchid] + LDA * coffA + roffA,
                                   LDA, Barray[batchid] + LDB * coffB + roffB,
                                   LDB, Carray[batchid] + LDC * coffC + roffC,
                                   LDC, alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_batched_rl_kernel(
    int M, int N,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC ,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    const int batchid = item_ct1.get_group(0);

    hemm_template_device_rl<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, Aarray[batchid] + LDA * coffA + roffA,
                                   LDA, Barray[batchid] + LDB * coffB + roffB,
                                   LDB, Carray[batchid] + LDC * coffC + roffC,
                                   LDC, alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_batched_ru_kernel(
    int M, int N,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC ,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    const int batchid = item_ct1.get_group(0);

    hemm_template_device_ru<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, Aarray[batchid] + LDA * coffA + roffA,
                                   LDA, Barray[batchid] + LDB * coffB + roffB,
                                   LDB, Carray[batchid] + LDC * coffC + roffC,
                                   LDC, alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
void hemm_template_batched(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, DIM, DIM);

    if( side == MagmaLeft ){
        if(uplo == MagmaLower){
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(n, BLK_N),
                                    magma_ceildiv(m, BLK_M));
                /*
                DPCT1049:847: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sA_acc_ct1(sycl::range<2>(BLK_M, BLK_M + 1), cgh);
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_batched_ll_kernel<T, DIM, BLK_M,
                                                                BLK_N, CONJA>(
                                    m, n, dA_array + i, ldda, dB_array + i,
                                    lddb, dC_array + i, lddc, alpha, beta,
                                    roffA, coffA, roffB, coffB, roffC, coffC,
                                    item_ct1, sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }else{
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(n, BLK_N),
                                    magma_ceildiv(m, BLK_M));

                /*
                DPCT1049:848: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sA_acc_ct1(sycl::range<2>(BLK_M, BLK_M + 1), cgh);
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_batched_lu_kernel<T, DIM, BLK_M,
                                                                BLK_N, CONJA>(
                                    m, n, dA_array + i, ldda, dB_array + i,
                                    lddb, dC_array + i, lddc, alpha, beta,
                                    roffA, coffA, roffB, coffB, roffC, coffC,
                                    item_ct1, sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }
    }else{
        if(uplo == MagmaLower){
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(n, BLK_N),
                                    magma_ceildiv(m, BLK_M));

                /*
                DPCT1049:849: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sA_acc_ct1(sycl::range<2>(BLK_N, BLK_N + 1), cgh);
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_batched_rl_kernel<T, DIM, BLK_M,
                                                                BLK_N, CONJA>(
                                    m, n, dA_array + i, ldda, dB_array + i,
                                    lddb, dC_array + i, lddc, alpha, beta,
                                    roffA, coffA, roffB, coffB, roffC, coffC,
                                    item_ct1, sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }else{
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(n, BLK_N),
                                    magma_ceildiv(m, BLK_M));

                /*
                DPCT1049:850: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sA_acc_ct1(sycl::range<2>(BLK_N, BLK_N + 1), cgh);
                        sycl::accessor<T, 2, sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_batched_ru_kernel<T, DIM, BLK_M,
                                                                BLK_N, CONJA>(
                                    m, n, dA_array + i, ldda, dB_array + i,
                                    lddb, dC_array + i, lddc, alpha, beta,
                                    roffA, coffA, roffB, coffB, roffC, coffC,
                                    item_ct1, sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }
    }
}
#endif //HEMM_TEMPLATE_KERNEL_BATCHED_CUH
