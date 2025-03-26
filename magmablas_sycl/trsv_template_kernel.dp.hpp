#include <sycl/sycl.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRSV_TEMPLATE_KERNEL_DP_HPP
#define TRSV_TEMPLATE_KERNEL_DP_HPP

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 

void trsv_template_NL_kernel(
        magma_diag_t diag, int n,
        T* dA,  int ldda,
        T* dx,  int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
    trsv_template_device_NL<T, NB>(diag, n, dA, ldda, dx, incx, item_ct1, sA,
                                   sx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static 

void trsv_template_NU_kernel(
        magma_diag_t diag, int n,
        T* dA,  int ldda,
        T* dx,  int incx , const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
    trsv_template_device_NU<T, NB>(diag, n, dA, ldda, dx, incx, item_ct1, sA,
                                   sx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 

void trsv_template_TL_kernel(
        magma_diag_t diag, int n,
        T* dA, int ldda,
        T* dx, int incx , const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
    trsv_template_device_TL<T, NB, CONJA>(diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA, sx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 

void trsv_template_TU_kernel(
        magma_diag_t diag, int n,
        T* dA, int ldda,
        T* dx, int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
    trsv_template_device_TU<T, NB, CONJA>(diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA, sx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB>
void trsv_small(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        T *dA, magma_int_t ldda,
        T *dx, magma_int_t incx, magma_queue_t queue )
{
    magma_int_t shape = 0;
    if      (uplo == MagmaLower  && transA == MagmaNoTrans   ) { shape = 0; } // NL
    else if (uplo == MagmaLower  && transA == MagmaTrans     ) { shape = 1; } // TL
    else if (uplo == MagmaLower  && transA == MagmaConjTrans ) { shape = 2; } // CL
    else if (uplo == MagmaUpper  && transA == MagmaNoTrans   ) { shape = 3; } // NU
    else if (uplo == MagmaUpper  && transA == MagmaTrans     ) { shape = 4; } // TU
    else if (uplo == MagmaUpper  && transA == MagmaConjTrans ) { shape = 5; } // CU

    sycl::range<3> threads(1, 1, NB);
    sycl::range<3> grid(1, 1, 1);

    switch(shape) {
        case 0: // NL
            {
                  ((sycl::queue *)(queue->sycl_stream()))
                      ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<T, 1> sA_acc_ct1(
                                sycl::range<1>(NB * NB), cgh);
                            sycl::local_accessor<T, 1> sx_acc_ct1(
                                sycl::range<1>(NB), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                      trsv_template_NL_kernel<T, NB>(
                                          diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sx_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get());
                                });
                      });
            }
            break;
        case 1: // TL
            {
                  ((sycl::queue *)(queue->sycl_stream()))
                      ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<T, 1> sA_acc_ct1(
                                sycl::range<1>(NB * NB), cgh);
                            sycl::local_accessor<T, 1> sx_acc_ct1(
                                sycl::range<1>(NB), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                      trsv_template_TL_kernel<T, NB, 0>(
                                          diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sx_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get());
                                });
                      });
            }
            break;
        case 2: // CL
            {
                  ((sycl::queue *)(queue->sycl_stream()))
                      ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<T, 1> sA_acc_ct1(
                                sycl::range<1>(NB * NB), cgh);
                            sycl::local_accessor<T, 1> sx_acc_ct1(
                                sycl::range<1>(NB), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                      trsv_template_TL_kernel<T, NB, 1>(
                                          diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sx_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get());
                                });
                      });
            }
            break;
        case 3: // NU
            {
                  ((sycl::queue *)(queue->sycl_stream()))
                      ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<T, 1> sA_acc_ct1(
                                sycl::range<1>(NB * NB), cgh);
                            sycl::local_accessor<T, 1> sx_acc_ct1(
                                sycl::range<1>(NB), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                      trsv_template_NU_kernel<T, NB>(
                                          diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sx_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get());
                                });
                      });
            }
            break;
        case 4: // TU
            {
                  ((sycl::queue *)(queue->sycl_stream()))
                      ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<T, 1> sA_acc_ct1(
                                sycl::range<1>(NB * NB), cgh);
                            sycl::local_accessor<T, 1> sx_acc_ct1(
                                sycl::range<1>(NB), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                      trsv_template_TU_kernel<T, NB, 0>(
                                          diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sx_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get());
                                });
                      });
            }
            break;
        case 5: // CU
            {
                  ((sycl::queue *)(queue->sycl_stream()))
                      ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<T, 1> sA_acc_ct1(
                                sycl::range<1>(NB * NB), cgh);
                            sycl::local_accessor<T, 1> sx_acc_ct1(
                                sycl::range<1>(NB), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                      trsv_template_TU_kernel<T, NB, 1>(
                                          diag, n, dA, ldda, dx, incx, item_ct1,
                                          sA_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sx_acc_ct1
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get());
                                });
                      });
            }
            break;
        default:; // propose something
    }
}

#endif //TRSV_TEMPLATE_KERNEL_DP_HPP
