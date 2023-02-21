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

#define PRECISION_d

#include "gemv_template_kernel_batched.dp.hpp"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v

// This is an internal routine, please see dgemv_batched.cpp for more details
extern "C" void
magmablas_dgemv_batched_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const double alpha,
    double const * const * dA_array, const double* dA, magma_int_t ldda, magma_int_t strideA,
    double const * const * dx_array, const double* dx, magma_int_t incx, magma_int_t stridex,
    const double beta,
    double** dy_array, double* dy, magma_int_t incy, magma_int_t stridey,
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
            if (m < n) { // Fat or square matrix
                if ( m <= 16) {
                    gemvn_template_batched<double, version(N, 72)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 32) {
                    gemvn_template_batched<double, version(N, 100)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( m <= 64) {
                    gemvn_template_batched<double, version(N, 122)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<double, version(N, 135)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            } else {   // Tall or square matrix
                if ( n <= 16) {
                    gemvn_template_batched<double, version(N, 128)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if ( n <= 64) {
                    gemvn_template_batched<double, version(N, 132)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<double, version(N, 135)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }
        else { // big size
            if (m < n) { // Fat matrix
                if (m <= 8) {
                    gemvn_template_batched<double, version(N, 79)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 16) {
                    gemvn_template_batched<double, version(N, 70)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 32) {
                    gemvn_template_batched<double, version(N, 104)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else if (m <= 32) {
                    gemvn_template_batched<double, version(N, 124)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<double, version(N, 135)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
            else { // (m > n) Tall matrix
                if (m <= 256) {
                    gemvn_template_batched<double, version(N, 137)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvn_template_batched<double, version(N, 140)>
                    ( m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }// big size
    }
    else {
        if (max(m, n) <= 96) { // small size
            if (m <= 16) {
                gemvc_template_batched<double, version(T, 42)>
                ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
            }
            else {
                gemvc_template_batched<double, version(T, 47)>
                ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
            }
        } else { // big size
            if (m <= n) { //  Fat or square matrix
                if (m <= 64) {
                    gemvc_template_batched<double, version(T, 47)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvc_template_batched<double, version(T, 91)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            } else { // (m > n) Tall matrix
                if (n <= 64) {
                    gemvc_template_batched<double, version(T, 90)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
                else {
                    gemvc_template_batched<double, version(T, 91)>
                    ( trans, m, n, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue );
                }
            }
        }
    }
}
