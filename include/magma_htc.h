/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

*/

#ifndef MAGMA_HTC_H
#define MAGMA_HTC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MAGMA mixed precision tensor cores function definitions
//
// In alphabetical order of base name (ignoring precision).

/* Half precision iterative refinement routines */
    
magma_int_t
magma_dhgesv_iteref_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_dsgesv_iteref_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_dxgesv_gmres_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_refinement_t facto_type, 
    magma_refinement_t solver_type,
    magma_int_t *iter,
    magma_int_t *info,
    real_Double_t *facto_time);

magma_int_t
magma_dfgmres_plu_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dLU_sprec, magma_int_t lddlusp,
    magmaDouble_ptr dLU_dprec, magma_int_t lddludp,
    magmaInt_ptr ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX, 
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t maxiter_inner, magma_int_t restrt_inner,
    magma_int_t userinitguess, 
    double tol, double innertol,
	double *rnorm0, magma_int_t *niters,
    magma_refinement_t solver_type,
    char *algoname, magma_int_t is_inner,
    magma_queue_t queue);

magma_int_t
magma_dsgelatrs_cpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr  dA, magma_int_t ldda,
    magmaInt_ptr        dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX,
    magma_int_t *info);

/* Half precision LU factorizations routines */

magma_int_t
magma_hgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info );

magma_int_t
magma_htgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info );

magma_int_t
magma_xhsgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info,
    magma_mp_type_t enable_tc,
    magma_mp_type_t mp_algo_type);

magma_int_t
magma_xshgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info,
    magma_mp_type_t enable_tc,
    magma_mp_type_t mp_algo_type);

magma_int_t 
    magma_get_hgetrf_nb( magma_int_t m, magma_int_t n );

magma_int_t 
magma_get_xgetrf_nb( 
        magma_int_t m, magma_int_t n, magma_int_t prev_nb, 
        magma_mp_type_t enable_tc, magma_mp_type_t mp_algo_type);

/* Half precision conversion routines */
void
magmablas_convert_dp2hp(
    magma_int_t m, magma_int_t n,
    const double  *dA, magma_int_t ldda,
    magmaHalf  *dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_convert_hp2dp(
    magma_int_t m, magma_int_t n,
    const magmaHalf  *dA, magma_int_t ldda,
    double  *dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_convert_hp2sp(
    magma_int_t m, magma_int_t n,
    const magmaHalf  *dA, magma_int_t ldda,
    float  *dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_convert_sp2hp(
    magma_int_t m, magma_int_t n,
    const float  *dA, magma_int_t ldda,
    magmaHalf  *dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_hlaswp(
    magma_int_t n,
    magmaHalf *dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

/* Eigenvalue refinement routines */
magma_int_t
magma_zcheevdx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_zcheevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dssyevdx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dssyevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

void magma_znormalize( magma_int_t n, magma_int_t m, magmaDoubleComplex *dX, magma_int_t ldx, magma_queue_t queue);
void magma_dnormalize( magma_int_t n, magma_int_t m, double *dX, magma_int_t ldx, magma_queue_t queue);
void magma_zdiag_scale( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                        double *x, magma_int_t incx,
                        magmaDoubleComplex *C, magma_int_t ldc, magma_queue_t queue );
void magma_ddiag_scale( magma_int_t m, magma_int_t n, double *A, magma_int_t lda,
                        double *x, magma_int_t incx,
                        double *C, magma_int_t ldc, magma_queue_t queue );
void magma_zsolve_sicesm( magma_int_t n, magma_int_t m, magmaDoubleComplex *dR, magma_int_t ldr,
                          magmaDoubleComplex *dC, double *dd, double* de, magmaDoubleComplex *df, double *dw,
                          magmaDoubleComplex *dwork, double *drwork, magma_queue_t queue );
void magma_dsolve_sicesm( magma_int_t n, magma_int_t m, double *dR, magma_int_t ldr,
                          double *dC, double *dd, double* de, double *df, double *dw,
                          double* dwork, double* drwork, magma_queue_t queue );
void magma_zupdate_eigenvalues( magma_int_t m, magmaDoubleComplex *dX, magma_int_t incx,
                                double *dw, magma_queue_t queue );
void magma_zadd_eigenvalues( magma_int_t m, magmaDoubleComplex *dX, magma_int_t incx,
                                                double *dw, magma_queue_t queue );
magma_int_t
magma_zchesicesm(
    magma_int_t n, magma_int_t m,
    magmaDoubleComplex *dA, magma_int_t lda,
    magmaFloatComplex *dfloatQ, magma_int_t ldq,
    magmaDoubleComplex *dX, magma_int_t ldx,
    double *w, double *d, double *e,
    magmaDoubleComplex *dwork, magma_int_t ldwork,
    magma_queue_t queue);

magma_int_t
magma_dssysicesm(
    magma_int_t n, magma_int_t m,
    double *dA, magma_int_t lda,
    float *dfloatQ, magma_int_t ldq,
    double *dX, magma_int_t ldx,
    double *w, double *d, double *e,
    double *dwork, magma_int_t ldwork,
    magma_queue_t queue);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_HTC_H */
