/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_s

#include "gemv_template_kernel_batched.dp.hpp"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v

// This is an internal routine, please see sgemv_batched.cpp for more details
extern "C" void
magmablas_sgemv_batched_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const float alpha,
    float const * const * dA_array, const float* dA, magma_int_t ldda, magma_int_t strideA,
    float const * const * dx_array, const float* dx, magma_int_t incx, magma_int_t stridex,
    const float beta,
    float** dy_array, float* dy, magma_int_t incy, magma_int_t stridey,
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
                if ( m <= 8) {
                    gemvn_template_batched<float, version(N, 32)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 16) {
                    gemvn_template_batched<float, version(N, 72)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 32) {
                    gemvn_template_batched<float, version(N, 97)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 64) {
                    gemvn_template_batched<float, version(N, 120)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<float, version(N, 130)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
            else {   // Tall matrix
                if ( n <= 16) {
                    gemvn_template_batched<float, version(N, 118)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<float, version(N, 120)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }
        else { // big size
            if (m < n) { // Fat matrix
                if (m <= 16) {
                    gemvn_template_batched<float, version(N, 79)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 32) {
                    gemvn_template_batched<float, version(N, 103)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 64) {
                    gemvn_template_batched<float, version(N, 126)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<float, version(N, 135)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
            else { // Tall or square matrix
                if (m <= 256) {
                    gemvn_template_batched<float, version(N, 137)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<float, version(N, 140)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }// big size
    }
    else {
        if (max(m, n) <= 96) { // small size
            gemvc_template_batched<float, version(T, 46)>
            ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
        }
        else { // big size
            if (m <= n) { //  Fat or square matrix
                if (m <= 64) {
                    gemvc_template_batched<float, version(T, 47)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvc_template_batched<float, version(T, 133)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
            else { // (m > n) Tall matrix
                if (n <= 8) {
                    gemvc_template_batched<float, version(T, 130)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvc_template_batched<float, version(T, 131)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }
    }
}
