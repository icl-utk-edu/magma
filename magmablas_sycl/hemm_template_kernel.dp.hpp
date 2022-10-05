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

#ifndef HEMM_TEMPLATE_KERNEL_CUH
#define HEMM_TEMPLATE_KERNEL_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "hemm_template_device.dp.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_ll_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta, sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    hemm_template_device_ll<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, A, LDA, B, LDB, C, LDC, alpha, beta,
                                   item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_lu_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta, sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    hemm_template_device_lu<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, A, LDA, B, LDB, C, LDC, alpha, beta,
                                   item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_rl_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta, sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    hemm_template_device_rl<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, A, LDA, B, LDB, C, LDC, alpha, beta,
                                   item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_ru_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta, sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    hemm_template_device_ru<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(M, N, A, LDA, B, LDB, C, LDC, alpha, beta,
                                   item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
void hemm_template(
    magma_side_t side, magma_uplo_t uplo, 
    magma_int_t m, magma_int_t n, 
    const T* dA, magma_int_t ldda,
    const T* dB, magma_int_t lddb,
          T* dC, magma_int_t lddc,
    T alpha, T beta, magma_queue_t queue)
{
    sycl::range<3> threads(1, DIM, DIM);
    sycl::range<3> grid(1, magma_ceildiv(n, BLK_N), magma_ceildiv(m, BLK_M));
    if( side == MagmaLeft ){
        if(uplo == MagmaLower){
            /*
            DPCT1049:851: The work-group size passed to the SYCL kernel may
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

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         hemm_template_ll_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                             m, n, dA, ldda, dB, lddb, dC, lddc,
                                             alpha, beta, item_ct1, sA_acc_ct1,
                                             sB_acc_ct1);
                                     });
                });
        }else{
            /*
            DPCT1049:852: The work-group size passed to the SYCL kernel may
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

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         hemm_template_lu_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                             m, n, dA, ldda, dB, lddb, dC, lddc,
                                             alpha, beta, item_ct1, sA_acc_ct1,
                                             sB_acc_ct1);
                                     });
                });
        }
    }else{
        if(uplo == MagmaLower){
            /*
            DPCT1049:853: The work-group size passed to the SYCL kernel may
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

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         hemm_template_rl_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                             m, n, dA, ldda, dB, lddb, dC, lddc,
                                             alpha, beta, item_ct1, sA_acc_ct1,
                                             sB_acc_ct1);
                                     });
                });
        }else{
            /*
            DPCT1049:854: The work-group size passed to the SYCL kernel may
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

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         hemm_template_ru_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                             m, n, dA, ldda, dB, lddb, dC, lddc,
                                             alpha, beta, item_ct1, sA_acc_ct1,
                                             sB_acc_ct1);
                                     });
                });
        }
    }
}
#endif //HEMM_TEMPLATE_KERNEL_CUH
