/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_c

#include "gemv_template_kernel_batched.cuh"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v

// This is an internal routine, please see cgemv_batched.cpp for more details
extern "C" void
magmablas_cgemv_batched_internal(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, const magmaFloatComplex* dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    magmaFloatComplex const * const * dx_array, const magmaFloatComplex* dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    const magmaFloatComplex beta,
    magmaFloatComplex** dy_array, magmaFloatComplex* dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( trans == MagmaNoTrans ) {
        if (max(m, n) <= 96) { // small size
            if (m < n) { // Fat matrix
                if ( m <= 16) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 70)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else if ( m <= 32) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 100)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else if ( m <= 64) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 117)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaFloatComplex, version(N, 131)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
            }
            else {  // Tall or square matrix
                if ( n <= 32) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 129)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaFloatComplex, version(N, 131)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
            }
        }
        else { // big size
            if (m < n) { // Fat matrix
                if (m <= 8) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 36)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else if (m <= 16) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 70)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else if (m <= 32) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 100)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else if (m <= 32) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 116)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaFloatComplex, version(N, 133)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
            }
            else { // Tall or square matrix
                if (m <= 256) {
                    gemvn_template_batched<magmaFloatComplex, version(N, 137)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaFloatComplex, version(N, 140)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
            }
        }// big size
    } else {
        if (max(m, n) <= 96) // small size
        {
            if (n <= 8) {
                gemvc_template_batched<magmaFloatComplex, version(T, 42)>
                ( trans, m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
            }
            else {
                gemvc_template_batched<magmaFloatComplex, version(T, 46)>
                ( trans, m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
            }
        }
        else // big size
        {
            if (m <= n) { //  Fat or square matrix
                if (m <= 64) {
                    gemvc_template_batched<magmaFloatComplex, version(T, 47)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else {
                    gemvc_template_batched<magmaFloatComplex, version(T, 90)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
            }
            else { // (m > n) Tall matrix
                if (n <= 8) {
                    gemvc_template_batched<magmaFloatComplex, version(T, 130)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
                else {
                    gemvc_template_batched<magmaFloatComplex, version(T, 90)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, Ai, Aj, dx_array, dx, incx, stridex, xi, beta, dy_array, dy, incy, stridey, yi, batchCount, queue );
                }
            }
        }
    }
}
