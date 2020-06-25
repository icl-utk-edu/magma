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
#include "gemm_template_device_defs.cuh"
#include "trmv_template_device.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
void trmv_small_template_kernel(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        int n, T* A, int ldda, T* X, int incx)
{
    trmv_small_template_device<T, NB, CONJA>
    (uplo, transA, diag, n, A, ldda, X, incx);
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

    dim3 threads(NB, 1, 1);
    dim3 grid( 1, 1, 1 );
    trmv_small_template_kernel<T, NB, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
    (uplo, transA, diag, n, dA, ldda, dX, incx);
}

#endif //TRMV_TEMPLATE_KERNEL_CUH
