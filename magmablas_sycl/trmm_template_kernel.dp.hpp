/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_KERNEL_CUH
#define TRMM_TEMPLATE_KERNEL_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "trmm_template_device.dp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 

void trmm_template_lNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    trmm_small_template_device_lNx<T, NB>(uplo, diag, m, n, alpha, A, ldda, B,
                                          lddb, item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 

void trmm_template_lTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    trmm_small_template_device_lTx<T, NB, CONJA>(
        uplo, diag, m, n, alpha, A, ldda, B, lddb, item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 

void trmm_template_rNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    trmm_small_template_device_rNx<T, NB>(uplo, diag, m, n, alpha, A, ldda, B,
                                          lddb, item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 

void trmm_template_rTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb, sycl::nd_item<3> item_ct1, T *sA, T *sB)
{
    trmm_small_template_device_rTx<T, NB, CONJA>(
        uplo, diag, m, n, alpha, A, ldda, B, lddb, item_ct1, sA, sB);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_lNx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    sycl::range<3> threads(1, NB, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, NB));
    /*
    DPCT1049:1442: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1>
            sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
        sycl::local_accessor<T, 1>
            sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             trmm_template_lNx_kernel<T, NB>(
                                 uplo, diag, m, n, alpha, dA, ldda, dB, lddb,
                                 item_ct1, sA_acc_ct1.get_pointer(),
                                 sB_acc_ct1.get_pointer());
                         });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_lTx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    sycl::range<3> threads(1, NB, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, NB));
    /*
    DPCT1049:1443: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1>
            sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
        sycl::local_accessor<T, 1>
            sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             trmm_template_lTx_kernel<T, NB, CONJA>(
                                 uplo, diag, m, n, alpha, dA, ldda, dB, lddb,
                                 item_ct1, sA_acc_ct1.get_pointer(),
                                 sB_acc_ct1.get_pointer());
                         });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_rNx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    sycl::range<3> threads(1, NB, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(m, NB));
    /*
    DPCT1049:1444: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1>
            sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
        sycl::local_accessor<T, 1>
            sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             trmm_template_rNx_kernel<T, NB>(
                                 uplo, diag, m, n, alpha, dA, ldda, dB, lddb,
                                 item_ct1, sA_acc_ct1.get_pointer(),
                                 sB_acc_ct1.get_pointer());
                         });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_rTx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    sycl::range<3> threads(1, NB, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(m, NB));
    /*
    DPCT1049:1445: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1>
            sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
        sycl::local_accessor<T, 1>
            sB_acc_ct1(sycl::range<1>(NB * NB), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             trmm_template_rTx_kernel<T, NB, CONJA>(
                                 uplo, diag, m, n, alpha, dA, ldda, dB, lddb,
                                 item_ct1, sA_acc_ct1.get_pointer(),
                                 sB_acc_ct1.get_pointer());
                         });
    });
}

#endif //TRMM_TEMPLATE_KERNEL_CUH
