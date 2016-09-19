/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef COMMONBLAS_Z_H
#define COMMONBLAS_Z_H

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Internal prototypes

// Tesla GEMM kernels
#define MAGMABLAS_ZGEMM( name ) \
void magmablas_zgemm_##name( \
    magmaDoubleComplex *C, const magmaDoubleComplex *A, const magmaDoubleComplex *B, \
    magma_int_t m, magma_int_t n, magma_int_t k, \
    magma_int_t lda, magma_int_t ldb, magma_int_t ldc, \
    magmaDoubleComplex alpha, magmaDoubleComplex beta )

MAGMABLAS_ZGEMM( a_0  );
MAGMABLAS_ZGEMM( ab_0 );
MAGMABLAS_ZGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_ZGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_ZGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_ZGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_ZGEMM( T_T_64_16_16_16_4_special );
MAGMABLAS_ZGEMM( T_T_64_16_16_16_4         );
                   
void magmablas_zgemm_tesla(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *B, magma_int_t ldb,
    magmaDoubleComplex beta,
    magmaDoubleComplex *C, magma_int_t ldc );

void magmablas_zgemv_tesla(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, magma_int_t incy );


// kernels used in dznrm2, zgeqr2x-v4, laqps2_gpu, zlarfbx, zlarfgx-v2, zlarfx
__global__ void
magma_zgemv_kernel1(int m, const magmaDoubleComplex * __restrict__ V, int ldv,
                    const magmaDoubleComplex * __restrict__ c,
                    magmaDoubleComplex *dwork);

__global__ void
magma_zgemv_kernel2(int m, int n, const magmaDoubleComplex * __restrict__ V, int ldv,
                    const magmaDoubleComplex * __restrict__ x, magmaDoubleComplex *c);

__global__ void
magma_zgemv_kernel3(int m, const magmaDoubleComplex * __restrict__ V, int ldv,
                    magmaDoubleComplex *c, magmaDoubleComplex *dwork,
                    magmaDoubleComplex *tau);

__global__ void
magma_ztrmv_tkernel(magmaDoubleComplex *T, int ldt, magmaDoubleComplex *v,
                                    magmaDoubleComplex *y);

__global__ void
magma_ztrmv_kernel2(const magmaDoubleComplex *T, int ldt,
                    magmaDoubleComplex *v, magmaDoubleComplex *y, magmaDoubleComplex *tau);

__global__ void
magma_dznrm2_adjust_kernel(double *xnorm, magmaDoubleComplex *c);


// kernels used in zhemv
__global__ void
zhemv_kernel_U(
    int n,
    magmaDoubleComplex const * __restrict__ A, int lda,
    magmaDoubleComplex const * __restrict__ x, int incx,
    magmaDoubleComplex       * __restrict__ work);

__global__ void
zhemv_kernel_U_sum(
    int n,
    magmaDoubleComplex alpha,
    int lda,
    magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy,
    magmaDoubleComplex const * __restrict__ work );

// kernels used in zsymv
__global__ void
zsymv_kernel_U(
    int n,
    magmaDoubleComplex const * __restrict__ A, int lda,
    magmaDoubleComplex const * __restrict__ x, int incx,
    magmaDoubleComplex       * __restrict__ work);

__global__ void
zsymv_kernel_U_sum(
    int n,
    magmaDoubleComplex alpha,
    int lda,
    magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy,
    magmaDoubleComplex const * __restrict__ work );

// kernels used in zhemv_mgpu
__global__ void
zhemv_kernel_U_mgpu(
    int n,
    magmaDoubleComplex const * __restrict__ A, int lda,
    magmaDoubleComplex const * __restrict__ x, int incx,
    magmaDoubleComplex       * __restrict__ work,
    int my_gpu_id,
    int ngpu,
    int block_offset );

__global__ void
zhemv_kernel_U_mgpu_sum(
    int n,
    magmaDoubleComplex alpha,
    int lda,
    magmaDoubleComplex       * __restrict__ y, int incy,
    magmaDoubleComplex const * __restrict__ work,
    int my_gpu_id,
    int ngpu,
    int block_offset);

#ifdef __cplusplus
}
#endif

#endif // COMMONBLAS_Z_H
