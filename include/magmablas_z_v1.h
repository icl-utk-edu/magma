/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef MAGMABLAS_Z_V1_H
#define MAGMABLAS_Z_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_types.h"
#include "magma_copy_v1.h"

#define MAGMA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_ztranspose_inplace_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void
magmablas_ztranspose_conj_inplace_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void
magmablas_ztranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat );

void
magmablas_ztranspose_conj_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat );

void
magmablas_zgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dAT,   magma_int_t ldda,
    magmaDoubleComplex          *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void
magmablas_zsetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr    dAT,   magma_int_t ldda,
    magmaDoubleComplex_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * RBT-related functions
   */
void
magmablas_zprbt_v1(
    magma_int_t n, 
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDoubleComplex_ptr du,
    magmaDoubleComplex_ptr dv );

void
magmablas_zprbt_mv_v1(
    magma_int_t n, 
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr db );

void
magmablas_zprbt_mtv_v1(
    magma_int_t n, 
    magmaDoubleComplex_ptr du,
    magmaDoubleComplex_ptr db );

  /*
   * Multi-GPU copy functions
   */
void
magma_zgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dA[], magma_int_t ldda,
    magmaDoubleComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_zsetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_zgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dA[], magma_int_t ldda,
    magmaDoubleComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_zsetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_zgeadd_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void
magmablas_zgeadd2_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void
magmablas_zlacpy_v1(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void
magmablas_zlacpy_conj_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, magma_int_t lda1,
    magmaDoubleComplex_ptr dA2, magma_int_t lda2 );

void
magmablas_zlacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void
magmablas_zlacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

double
magmablas_zlange_v1(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork );

double
magmablas_zlanhe_v1(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork );

double
magmablas_zlansy_v1(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork );

void
magmablas_zlarfg_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dalpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dtau );

void
magmablas_zlascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_zlascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaDoubleComplex_const_ptr dW, magma_int_t lddw,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_zlascl2_v1(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_zlascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_zlaset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void
magmablas_zlaset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void
magmablas_zlaswp_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_zlaswp2_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci );

void
magmablas_zlaswp_sym_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_zlaswpx_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_zsymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void
magmablas_zsymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void
magmablas_ztrtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr d_dinvA );

  /*
   * to cleanup (alphabetical order)
   */
void
magmablas_dznrm2_adjust_v1(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDoubleComplex_ptr dc );

void
magmablas_dznrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc );

void
magmablas_dznrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm );

void
magmablas_dznrm2_row_check_adjust_v1(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc );

magma_int_t
magma_zlarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT, magma_int_t lddt,
    magmaDoubleComplex_ptr dC,       magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_zlarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT, magma_int_t lddt,
    magmaDoubleComplex_ptr dC,       magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,    magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt,  magma_int_t ldworkvt );

void
magma_zlarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr dT, magma_int_t ldt,
    magmaDoubleComplex_ptr c,
    magmaDoubleComplex_ptr dwork );

void
magma_zlarfg_gpu_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dAkk );

void
magma_zlarfgtx_gpu_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr T,  magma_int_t ldt,
    magmaDoubleComplex_ptr dwork );

void
magma_zlarfgx_gpu_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter );

void
magma_zlarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr tau,
    magmaDoubleComplex_ptr C,  magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDoubleComplex_ptr dT, magma_int_t iter,
    magmaDoubleComplex_ptr work );


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_zaxpycp_v1(
    magma_int_t m,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_const_ptr db );

void
magmablas_zswap_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy );

void
magmablas_zswapblk_v1(
    magma_order_t order,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void
magmablas_zswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
magmablas_zgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void
magmablas_zgemv_conj_v1(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy );

magma_int_t
magmablas_zhemv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

magma_int_t
magmablas_zsymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_zgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zhemm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zsymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zsyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zher2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double  beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zsyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_zherk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double  alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double  beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magmablas_ztrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void
magmablas_ztrsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length );

void
magmablas_ztrsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length );


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// =============================================================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_zsetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        magma_zsetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_zgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        magma_zgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_zcopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        magma_zcopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
magma_zsetvector_v1_internal(
    magma_int_t n,
    magmaDoubleComplex const    *hx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_v1_internal( n, sizeof(magmaDoubleComplex),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
magma_zgetvector_v1_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex          *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_v1_internal( n, sizeof(magmaDoubleComplex),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
magma_zcopyvector_v1_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_v1_internal( n, sizeof(magmaDoubleComplex),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

#define magma_zsetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        magma_zsetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_zgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        magma_zgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define magma_zcopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_zcopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
magma_zsetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const    *hA_src, magma_int_t lda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_setmatrix_v1_internal( m, n, sizeof(magmaDoubleComplex),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
magma_zgetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex          *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    magma_getmatrix_v1_internal( m, n, sizeof(magmaDoubleComplex),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
magma_zcopymatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_copymatrix_v1_internal( m, n, sizeof(magmaDoubleComplex),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_izamax_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t
magma_izamin_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
double
magma_dzasum_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

void
magma_zaxpy_v1(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void
magma_zcopy_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotc_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotu_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double
magma_dznrm2_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

void
magma_zrot_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, magmaDoubleComplex ds );

void
magma_zdrot_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, double ds );

#ifdef MAGMA_REAL
void
magma_zrotm_v1(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    magmaDouble_const_ptr param );

void
magma_zrotmg_v1(
    magmaDouble_ptr d1, magmaDouble_ptr       d2,
    magmaDouble_ptr x1, magmaDouble_const_ptr y1,
    magmaDouble_ptr param );
#endif  // MAGMA_REAL

void
magma_zscal_v1(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx );

void
magma_zdscal_v1(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx );

void
magma_zswap_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
magma_zgemv_v1(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void
magma_zgerc_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void
magma_zgeru_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void
magma_zhemv_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void
magma_zher_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void
magma_zher2_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void
magma_ztrmv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx );

void
magma_ztrsv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
magma_zgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_zsymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_zhemm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_zsyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_zher2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_zsyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_zherk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void
magma_ztrmm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void
magma_ztrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef MAGMA_COMPLEX

#endif // MAGMABLAS_Z_H
