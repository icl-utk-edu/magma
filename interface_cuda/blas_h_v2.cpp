/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/
#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// =============================================================================
// Level 1 BLAS

// =============================================================================
// Level 2 BLAS

// =============================================================================
// Level 3 BLAS

/***************************************************************************//**
    Perform FP16 matrix-matrix product, \f$ C = \alpha op(A) op(B) + \beta C \f$.
    This routine requires CUDA 7.5 or greater. 

    @param[in]
    transA  Operation op(A) to perform on matrix A.

    @param[in]
    transB  Operation op(B) to perform on matrix B.

    @param[in]
    m       Number of rows of C and op(A). m >= 0.

    @param[in]
    n       Number of columns of C and op(B). n >= 0.

    @param[in]
    k       Number of columns of op(A) and rows of op(B). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      HALF PRECISION array on GPU device.
            If transA == MagmaNoTrans, the m-by-k matrix A of dimension (ldda,k), ldda >= max(1,m); \n
            otherwise,                 the k-by-m matrix A of dimension (ldda,m), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      HALF PRECISION array on GPU device.
            If transB == MagmaNoTrans, the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k); \n
            otherwise,                 the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      HALF PRECISION array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemm
*******************************************************************************/
extern "C" void
magma_hgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf_const_ptr dA, magma_int_t ldda,
    magmaHalf_const_ptr dB, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
#if CUDA_VERSION >= 7500
    magma_int_t arch = magma_getdevice_arch();
    if( arch >= 530 ) {
        #if CUDA_VERSION >= 9000
        // turn on tensor cores by default
        cublasSetMathMode(queue->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
        #endif

        cublasHgemm(
            queue->cublas_handle(),
            cublas_trans_const( transA ),
            cublas_trans_const( transB ),
            int(m), int(n), int(k),
            &alpha, dA, int(ldda),
                    dB, int(lddb),
            &beta,  dC, int(lddc) );
        
        #if CUDA_VERSION >= 9000
        // roll back to default
        cublasSetMathMode(queue->cublas_handle(), CUBLAS_DEFAULT_MATH);
        #endif
    }
    else {
        printf("ERROR: unsupported architecture for %s \n", __func__ );
    }
#else
    printf("ERROR: unsupported CUDA version for %s \n", __func__ );
#endif    // CUDA_VERSION >= 7500
}
#endif // HAVE_CUBLAS

