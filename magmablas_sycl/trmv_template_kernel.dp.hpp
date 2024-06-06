/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRMV_TEMPLATE_KERNEL_CUH
#define TRMV_TEMPLATE_KERNEL_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "trmv_template_device.dp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static 
void trmv_small_template_kernel(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        int n, T* A, int ldda, T* X, int incx, sycl::nd_item<3> item_ct1, T *sA,
        T *sX)
{
    trmv_small_template_device<T, NB, CONJA>(uplo, transA, diag, n, A, ldda, X,
                                             incx, item_ct1, sA, sX);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, int CONJA>
void trmv_template(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n, T* dA, magma_int_t ldda, T* dX, magma_int_t incx,
    magma_queue_t queue)
{
    if(!(transA == MagmaNoTrans)) {
        // the device code transposes the matrix in shared memory
        // so we should switch the uplo Trans and ConjTrans
        uplo = (uplo == MagmaLower) ? MagmaUpper : MagmaLower;
    }

    sycl::range<3> threads(1, 1, NB);
    sycl::range<3> grid(1, 1, 1);
    // From trmv_small_template_device:
    // const int slda = NB+1;
    // used this definition in creation of shared memory accessor here
    // (originally added as `slda` by dpct; see CUDA version of
    // device function for reference)
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1>
            sA_acc_ct1(sycl::range<1>(/*slda*/(NB+1) * NB), cgh);
        sycl::local_accessor<T, 1>
            sX_acc_ct1(sycl::range<1>(NB), cgh);

  	cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             trmv_small_template_kernel<T, NB, CONJA>(
                                 uplo, transA, diag, n, dA, ldda, dX, incx,
                                 item_ct1, sA_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get(),
				 sX_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

#endif //TRMV_TEMPLATE_KERNEL_CUH
