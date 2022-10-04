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
void trmm_template_vbatched_lNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];

    // check if offsets produce out-of-bound pointers
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;

    // compute the maximum allowed value for m, n based on the input offsets
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_n, NB)) return;

    trmm_small_template_device_lNx<T, NB>(
        uplo, diag, my_m, my_n, alpha,
        Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
        Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid],
        item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 
void trmm_template_vbatched_lTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];

    // check if offsets produce out-of-bound pointers
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;

    // compute the maximum allowed value for m, n based on the input offsets
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;

    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_n, NB)) return;
    trmm_small_template_device_lTx<T, NB, CONJA>(
        uplo, diag, my_m, my_n, alpha,
        Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
        Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid],
        item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 
void trmm_template_vbatched_rNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];

    // check if offsets produce out-of-bound pointers
    if( my_n < Ai || my_n < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;

    // compute the maximum allowed value for m, n based on the input offsets
    my_n -= max( Bj, max( Ai, Aj ) );
    my_m -= Bi;

    // check if the user forces values for m, n, and k
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_m, NB)) return;
    trmm_small_template_device_rNx<T, NB>(
        uplo, diag, my_m, my_n, alpha,
        Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
        Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid],
        item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 
void trmm_template_vbatched_rTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t *m, magma_int_t *n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t *ldda,
                 T** Barray, int Bi, int Bj, magma_int_t *lddb,
        int max_m, int max_n, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_n < Ai || my_n < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;
    // compute the maximum allowed value for m, n based on the input offsets
    my_n -= max( Bj, max( Ai, Aj ) );
    my_m -= Bi;

    // check if the user forces values for m, n, and k
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );

    if(my_m <= 0 || my_n <= 0) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_m, NB)) return;
    trmm_small_template_device_rTx<T, NB, CONJA>(
        uplo, diag, my_m, my_n, alpha,
        Aarray[batchid] + (int)ldda[batchid] * Aj + Ai, (int)ldda[batchid],
        Barray[batchid] + (int)lddb[batchid] * Bj + Bi, (int)lddb[batchid],
        item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_vbatched_lNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_n, NB));
        /*
        DPCT1049:1446: The work-group size passed to the SYCL kernel may exceed
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
                        trmm_template_vbatched_lNx_kernel<T, NB>(
                            uplo, diag, m + i, n + i, alpha, dA_array + i, Ai,
                            Aj, ldda + i, dB_array + i, Bi, Bj, lddb + i, max_m,
                            max_n, item_ct1, sA_acc_ct1.get_pointer(),
                            sB_acc_ct1.get_pointer());
                    });
            });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_vbatched_lTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_n, NB));
        /*
        DPCT1049:1447: The work-group size passed to the SYCL kernel may exceed
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
                        trmm_template_vbatched_lTx_kernel<T, NB, CONJA>(
                            uplo, diag, m + i, n + i, alpha, dA_array + i, Ai,
                            Aj, ldda + i, dB_array + i, Bi, Bj, lddb + i, max_m,
                            max_n, item_ct1, sA_acc_ct1.get_pointer(),
                            sB_acc_ct1.get_pointer());
                    });
            });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_vbatched_rNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_m, NB));
        /*
        DPCT1049:1448: The work-group size passed to the SYCL kernel may exceed
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
                        trmm_template_vbatched_rNx_kernel<T, NB>(
                            uplo, diag, m + i, n + i, alpha, dA_array + i, Ai,
                            Aj, ldda + i, dB_array + i, Bi, Bj, lddb + i, max_m,
                            max_n, item_ct1, sA_acc_ct1.get_pointer(),
                            sB_acc_ct1.get_pointer());
                    });
            });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_vbatched_rTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, NB, NB);
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_m, NB));
        /*
        DPCT1049:1449: The work-group size passed to the SYCL kernel may exceed
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
                        trmm_template_vbatched_rTx_kernel<T, NB, CONJA>(
                            uplo, diag, m + i, n + i, alpha, dA_array + i, Ai,
                            Aj, ldda + i, dB_array + i, Bi, Bj, lddb + i, max_m,
                            max_n, item_ct1, sA_acc_ct1.get_pointer(),
                            sB_acc_ct1.get_pointer());
                    });
            });
    }
}
#endif //TRMM_TEMPLATE_KERNEL_BATCHED_CUH
