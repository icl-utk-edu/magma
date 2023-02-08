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

#define PRECISION_z

#include "gemv_template_kernel_batched.dp.hpp"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v

// This is an internal routine, please see zgemv_batched.cpp for more details
extern "C" void
magmablas_zgemv_batched_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA_array[], const magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    const magmaDoubleComplex* dx_array[], const magmaDoubleComplex* dx, magma_int_t incx, magma_int_t stridex,
    const magmaDoubleComplex beta,
    magmaDoubleComplex* dy_array[], magmaDoubleComplex* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    if ( trans == MagmaNoTrans ) {
        if (max(m, n) <= 96) { // small size
            if (m < n) { // Fat matrix
                if ( m <= 8) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 72)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 32) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 100)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 64) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 121)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 132)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            } else {   // Tall or square matrix
                if ( n <= 16) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 129)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( n <= 64) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 131)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 132)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }
        else { // big size
            if (m < n) { // Fat matrix
                if (m <= 16) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 72)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 32) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 100)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 64) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 116)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 133)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
            else { // Tall or square matrix
                if (m <= 256) {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 137)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 140)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }// big size
    } else {
        if (max(m, n) <= 96) { // small size
            if (n <= 16) {
                gemvc_template_batched<magmaDoubleComplex, version(T, 42)>
                ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
            }
            else {
                gemvc_template_batched<magmaDoubleComplex, version(T, 46)>
                ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
            }
        }
        else { // big size
            if (m <= n) { // Fat or square matrix
                if (m <= 64) {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 47)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 46)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
            else// (m > n) Tall matrix
            {
                if (n <= 8) {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 130)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 46)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }
    }
}
