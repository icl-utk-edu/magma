/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef MAGMA_Z_H
#define MAGMA_Z_H

#include "magma_types.h"
#include "magma_zgehrd_m.h"

#define MAGMA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MAGMA Auxiliary functions to get the NB used

#ifdef MAGMA_REAL
magma_int_t magma_get_dlaex3_m_nb();       // defined in dlaex3_m.cpp
#endif

// Cholesky, LU, symmetric indefinite
magma_int_t magma_get_zpotrf_nb( magma_int_t n );
magma_int_t magma_get_zgetrf_nb( magma_int_t m, magma_int_t n );
magma_int_t magma_get_zgetrf_native_nb( magma_int_t m, magma_int_t n );
magma_int_t magma_get_zgetri_nb( magma_int_t n );
magma_int_t magma_get_zhetrf_nb( magma_int_t n );
magma_int_t magma_get_zhetrf_nopiv_nb( magma_int_t n );
magma_int_t magma_get_zhetrf_aasen_nb( magma_int_t n );

// QR
magma_int_t magma_get_zgeqp3_nb( magma_int_t m, magma_int_t n );
magma_int_t magma_get_zgeqrf_nb( magma_int_t m, magma_int_t n );
magma_int_t magma_get_zgeqlf_nb( magma_int_t m, magma_int_t n );
magma_int_t magma_get_zgelqf_nb( magma_int_t m, magma_int_t n );

// eigenvalues
magma_int_t magma_get_zgehrd_nb( magma_int_t n );
magma_int_t magma_get_zhetrd_nb( magma_int_t n );
magma_int_t magma_get_zhegst_nb( magma_int_t n );
magma_int_t magma_get_zhegst_m_nb( magma_int_t n );

// SVD
magma_int_t magma_get_zgebrd_nb( magma_int_t m, magma_int_t n );
magma_int_t magma_get_zgesvd_nb( magma_int_t m, magma_int_t n );

// 2-stage eigenvalues
magma_int_t magma_get_zbulge_nb( magma_int_t n, magma_int_t nbthreads );
magma_int_t magma_get_zbulge_nb_mgpu( magma_int_t n );
magma_int_t magma_get_zbulge_vblksiz( magma_int_t n, magma_int_t nb, magma_int_t nbthreads );
magma_int_t magma_get_zbulge_gcperf();


// =============================================================================
// MAGMA function definitions
//
// In alphabetical order of base name (ignoring precision).
// Keep different versions of the same routine together, sorted this way:
// cpu (no suffix), gpu (_gpu), cpu/multi-gpu (_m), multi-gpu (_mgpu). Ex:
// magma_zheevdx
// magma_zheevdx_gpu
// magma_zheevdx_m
// magma_zheevdx_2stage
// magma_zheevdx_2stage_m

#ifdef MAGMA_REAL
// only applicable to real [sd] precisions
void
magma_dmove_eig(
    magma_range_t range, magma_int_t n, double *w,
    magma_int_t *il, magma_int_t *iu, double vl, double vu, magma_int_t *mout);

// defined in dlaex3.cpp
void
magma_zvrange(
    magma_int_t k, double *d, magma_int_t *il, magma_int_t *iu, double vl, double vu);

void
magma_zirange(
    magma_int_t k, magma_int_t *indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu);
#endif  // MAGMA_REAL

// ------------------------------------------------------------ zge routines
magma_int_t
magma_zgebrd(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *d, double *e,
    magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    #ifdef MAGMA_COMPLEX
    magmaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    magmaDoubleComplex *VL, magma_int_t ldvl,
    magmaDoubleComplex *VR, magma_int_t ldvr,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    #ifdef MAGMA_COMPLEX
    magmaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    magmaDoubleComplex *VL, magma_int_t ldvl,
    magmaDoubleComplex *VR, magma_int_t ldvr,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dwork, magmaDoubleComplex *work,
    magma_int_t *info);

magma_int_t
magma_zgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex *T,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgehrd2(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgelqf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgelqf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgels(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr A, magma_int_t lda,
    magmaDoubleComplex_ptr B, magma_int_t ldb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zggrqf(
    magma_int_t m, magma_int_t p, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *taua,
    magmaDoubleComplex *B, magma_int_t ldb,
    magmaDoubleComplex *taub,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgglse(
    magma_int_t m, magma_int_t n, magma_int_t p,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magmaDoubleComplex *c, magmaDoubleComplex *d, magmaDoubleComplex *x,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgels3_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgeqlf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqp3(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqp3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dwork, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dwork,
    magma_queue_t queue,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqr2x_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr        dwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqr2x2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr        dwork,
    magma_int_t *info);

magma_int_t
magma_zgeqr2x3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr        dwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr        dwork,
    magma_queue_t queue,
    magma_int_t *info);

magma_int_t
magma_zgeqrf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqrf_m(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqrf_ooc(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magma_int_t *info);

magma_int_t
magma_zgeqrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magmaDoubleComplex *tau,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqrf3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info);

magma_int_t
magma_zgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex const *tau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgeqrs3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex const *tau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgerbt_gpu(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *U, magmaDoubleComplex *V,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgerfs_nopiv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex_ptr dworkd, magmaDoubleComplex_ptr dAF,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_zgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *s,
    magmaDoubleComplex *U, magma_int_t ldu,
    magmaDoubleComplex *VT, magma_int_t ldvt,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info);

magma_int_t
magma_zgesv(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_zgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_zgesv_nopiv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgesv_rbt(
    magma_bool_t ref, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_zgesvd(
    magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda, double *s,
    magmaDoubleComplex *U,    magma_int_t ldu,
    magmaDoubleComplex *VT,   magma_int_t ldvt,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_queue_t queue,
    magma_int_t *info);

magma_int_t 
magma_zgetf2_native_fused( 
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magma_int_t *ipiv, magma_int_t gbstep, 
    magma_int_t *flags, 
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zgetf2_native(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *dipiv, magma_int_t* dipivinfo, 
    magma_int_t *dinfo, magma_int_t gbstep, 
    magma_queue_t queue, magma_queue_t update_queue);

// CUDA MAGMA only
magma_int_t
magma_zgetf2_nopiv(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zgetrf_recpanel_native(
    magma_int_t m, magma_int_t n,    
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t* dipiv, magma_int_t* dipivinfo,
    magma_int_t *dinfo, magma_int_t gbstep, 
    magma_queue_t queue, magma_queue_t update_queue );

magma_int_t
magma_zgetrf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_zgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_zgetrf_native(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info );

// CUDA MAGMA only
magma_int_t
magma_zgetrf_m(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_zgetrf_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgetrf2(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_zgetrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
    magmaDoubleComplex_ptr d_lAT[], magma_int_t lddat,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr d_lAP[],
    magmaDoubleComplex *W, magma_int_t ldw,
    magma_queue_t queues[][2],
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgetrf_nopiv(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgetrf_nopiv_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_zgetri_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zgetrs_nopiv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// ------------------------------------------------------------ zhe routines
magma_int_t
magma_zheevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevd_gpu(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *w,
    magmaDoubleComplex *wA,  magma_int_t ldwa,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevd_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevdx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevdx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *wA,  magma_int_t ldwa,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevdx_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevdx_2stage_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef MAGMA_COMPLEX
// no real [sd] precisions available
// CUDA MAGMA only
magma_int_t
magma_zheevr(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout,
    double *w,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magma_int_t *isuppz,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevr_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout,
    double *w,
    magmaDoubleComplex_ptr dZ, magma_int_t lddz,
    magma_int_t *isuppz,
    magmaDoubleComplex *wA, magma_int_t ldwa,
    magmaDoubleComplex *wZ, magma_int_t ldwz,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout,
    double *w,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zheevx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    double abstol, magma_int_t *mout,
    double *w,
    magmaDoubleComplex_ptr dZ, magma_int_t lddz,
    magmaDoubleComplex *wA, magma_int_t ldwa,
    magmaDoubleComplex *wZ, magma_int_t ldwz,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // MAGMA_COMPLEX

// CUDA MAGMA only
magma_int_t
magma_zhegst(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegst_gpu(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegst_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegvd(
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double *w, magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegvd_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegvdx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegvdx_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegvdx_2stage(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhegvdx_2stage_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *mout, double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef MAGMA_COMPLEX
// no real [sd] precisions available
// CUDA MAGMA only
magma_int_t
magma_zhegvr(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    double abstol, magma_int_t *mout, double *w,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magma_int_t *isuppz, magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// no real [sd] precisions available
// CUDA MAGMA only
magma_int_t
magma_zhegvx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    double abstol, magma_int_t *mout, double *w,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magmaDoubleComplex *work, magma_int_t lwork, double *rwork,
    magma_int_t *iwork, magma_int_t *ifail,
    magma_int_t *info);
#endif

magma_int_t
magma_zhesv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhesv_nopiv_gpu(
    magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_zhetrd(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhetrd_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *wA,  magma_int_t ldwa,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhetrd2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *wA,  magma_int_t ldwa,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dwork, magma_int_t ldwork,
    magma_int_t *info);

// TODO: rename magma_zhetrd_m?
// CUDA MAGMA only
magma_int_t
magma_zhetrd_mgpu(
    magma_int_t ngpu, magma_int_t nqueue,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhetrd_hb2st(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    magmaDoubleComplex *A, magma_int_t lda,
    double *d, double *e,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU, magma_int_t compT,
    magmaDoubleComplex *T, magma_int_t ldt);

// CUDA MAGMA only
magma_int_t
magma_zhetrd_he2hb(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhetrd_he2hb_mgpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dAmgpu[], magma_int_t ldda,
    magmaDoubleComplex_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_zhetrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhetrf_aasen(
    magma_uplo_t uplo, magma_int_t cpu_panel, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv, magma_int_t *info);

magma_int_t
magma_zhetrf_nopiv(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zhetrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zhetrf_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zhetrs_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// ------------------------------------------------------------ [dz]la routines
#ifdef MAGMA_REAL
// only applicable to real [sd] precisions
magma_int_t
magma_dlaex0(
    magma_int_t n, double *d, double *e,
    double *Q, magma_int_t ldq,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_dlaex0_m(
    magma_int_t ngpu,
    magma_int_t n, double *d, double *e,
    double *Q, magma_int_t ldq,
    double *work, magma_int_t *iwork,
    magma_range_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_dlaex1(
    magma_int_t n, double *d,
    double *Q, magma_int_t ldq,
    magma_int_t *indxq, double rho, magma_int_t cutpnt,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_dlaex1_m(
    magma_int_t ngpu,
    magma_int_t n, double *d,
    double *Q, magma_int_t ldq,
    magma_int_t *indxq, double rho, magma_int_t cutpnt,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t
magma_dlaex3(
    magma_int_t k, magma_int_t n, magma_int_t n1, double *d,
    double *Q, magma_int_t ldq,
    double rho,
    double *dlamda, double *Q2, magma_int_t *indx,
    magma_int_t *ctot, double *w, double *s, magma_int_t *indxq,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_dlaex3_m(
    magma_int_t ngpu,
    magma_int_t k, magma_int_t n, magma_int_t n1, double *d,
    double *Q, magma_int_t ldq, double rho,
    double *dlamda, double *Q2, magma_int_t *indx,
    magma_int_t *ctot, double *w, double *s, magma_int_t *indxq,
    magmaDouble_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);
#endif  // MAGMA_REAL

magma_int_t
magma_zlabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex     *A, magma_int_t lda,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *d, double *e, magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
    magmaDoubleComplex     *X, magma_int_t ldx,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex     *Y, magma_int_t ldy,
    magmaDoubleComplex_ptr dY, magma_int_t lddy,
    magmaDoubleComplex  *work, magma_int_t lwork,
    magma_queue_t queue);

magma_int_t
magma_zlahef_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex    *hA, magma_int_t lda,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dW, magma_int_t lddw,
    magma_queue_t queues[], magma_event_t event[],
    magma_int_t *info);

magma_int_t
magma_zlahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dV, magma_int_t lddv,
    magmaDoubleComplex *A,  magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T,  magma_int_t ldt,
    magmaDoubleComplex *Y,  magma_int_t ldy,
    magma_queue_t queue);

// CUDA MAGMA only
magma_int_t
magma_zlahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t ldt,
    magmaDoubleComplex *Y, magma_int_t ldy,
    struct zgehrd_data *data);

magma_int_t
magma_zlahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex     *A, magma_int_t lda,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dY, magma_int_t lddy,
    magmaDoubleComplex_ptr dV, magma_int_t lddv,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue);

// CUDA MAGMA only
magma_int_t
magma_zlahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    struct zgehrd_data *data);

#ifdef MAGMA_REAL
// CUDA MAGMA only
magma_int_t
magma_dlaln2(
    magma_int_t trans, magma_int_t na, magma_int_t nw,
    double smin, double ca, const double *A, magma_int_t lda,
    double d1, double d2,   const double *B, magma_int_t ldb,
    double wr, double wi, double *X, magma_int_t ldx,
    double *scale, double *xnorm,
    magma_int_t *info);
#endif

// CUDA MAGMA only
magma_int_t
magma_zlaqps(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex *A,  magma_int_t lda,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, magmaDoubleComplex *tau, double *vn1, double *vn2,
    magmaDoubleComplex *auxv,
    magmaDoubleComplex *F,  magma_int_t ldf,
    magmaDoubleComplex_ptr dF, magma_int_t lddf);

// CUDA MAGMA only
magma_int_t
magma_zlaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    double *vn1, double *vn2,
    magmaDoubleComplex_ptr dauxv,
    magmaDoubleComplex_ptr dF, magma_int_t lddf);

// CUDA MAGMA only
magma_int_t
magma_zlaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr dvn1, magmaDouble_ptr dvn2,
    magmaDoubleComplex_ptr dauxv,
    magmaDoubleComplex_ptr dF,  magma_int_t lddf,
    magmaDouble_ptr dlsticcs,
    magma_queue_t queue);

#ifdef MAGMA_REAL
// CUDA MAGMA only
magma_int_t
magma_zlaqtrsd(
    magma_trans_t trans, magma_int_t n,
    const double *T, magma_int_t ldt,
    double *x,       magma_int_t ldx,
    const double *cnorm,
    magma_int_t *info);
#endif

// CUDA MAGMA only
magma_int_t
magma_zlarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaDoubleComplex_const_ptr dv, magmaDoubleComplex_const_ptr dtau,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magma_queue_t queue);

// magma_zlarfb_gpu
// see magmablas_q.h

// in zgeqr2x_gpu-v3.cpp
// CUDA MAGMA only
magma_int_t
magma_zlarfb2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT, magma_int_t lddt,
    magmaDoubleComplex_ptr dC,       magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,    magma_int_t ldwork,
    magma_queue_t queue);

magma_int_t
magma_zlatrd(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *W, magma_int_t ldw,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dW, magma_int_t lddw,
    magma_queue_t queue);

// CUDA MAGMA only
magma_int_t
magma_zlatrd2(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *A,  magma_int_t lda,
    double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *W,  magma_int_t ldw,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dW, magma_int_t lddw,
    magmaDoubleComplex_ptr dwork, magma_int_t ldwork,
    magma_queue_t queue);

// CUDA MAGMA only
magma_int_t
magma_zlatrd_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb, magma_int_t nb0,
    magmaDoubleComplex *A,  magma_int_t lda,
    double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex    *W,       magma_int_t ldw,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaDoubleComplex_ptr dW[],    magma_int_t lddw,
    magmaDoubleComplex    *hwork,   magma_int_t lhwork,
    magmaDoubleComplex_ptr dwork[], magma_int_t ldwork,
    magma_queue_t queues[]);

#ifdef MAGMA_COMPLEX
// CUDA MAGMA only
magma_int_t
magma_zlatrsd(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_diag_t diag, magma_bool_t normin,
    magma_int_t n, const magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex lambda,
    magmaDoubleComplex *x,
    double *scale, double *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_zlauum(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zlauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

// ------------------------------------------------------------ zpo routines
magma_int_t
magma_zposv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_zposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info);

magma_int_t
magma_zpotrf_rectile_native(
    magma_uplo_t uplo, magma_int_t n, magma_int_t recnb,    
    magmaDoubleComplex* dA,    magma_int_t ldda, magma_int_t gbstep, 
    magma_int_t *dinfo,  magma_int_t *info, magma_queue_t queue);

magma_int_t
magma_zpotrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_zpotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

// CUDA MAGMA only
magma_int_t
magma_zpotrf_m(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zpotrf_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zpotrf_mgpu_right(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zpotrf3_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magmaDoubleComplex_ptr d_lP[], magma_int_t lddp,
    magmaDoubleComplex *A, magma_int_t lda, magma_int_t h,
    magma_queue_t queues[][3], magma_event_t events[][5],
    magma_int_t *info);

magma_int_t
magma_zpotri(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_zpotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_zpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// ------------------------------------------------------------ zsy routines
#ifdef MAGMA_COMPLEX
// CUDA MAGMA only
magma_int_t
magma_zsysv_nopiv_gpu(
    magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zsytrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zsytrf_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zsytrs_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);
#endif

// ------------------------------------------------------------ zst routines
magma_int_t
magma_zstedx(
    magma_range_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double *d, double *e,
    magmaDoubleComplex *Z, magma_int_t ldz,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magmaDouble_ptr dwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zstedx_m(
    magma_int_t ngpu,
    magma_range_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double *d, double *e,
    magmaDoubleComplex *Z, magma_int_t ldz,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// ------------------------------------------------------------ ztr routines
// CUDA MAGMA only
magma_int_t
magma_ztrevc3(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    magmaDoubleComplex *T,  magma_int_t ldt,
    magmaDoubleComplex *VL, magma_int_t ldvl,
    magmaDoubleComplex *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_ztrevc3_mt(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    magmaDoubleComplex *T,  magma_int_t ldt,
    magmaDoubleComplex *VL, magma_int_t ldvl,
    magmaDoubleComplex *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef MAGMA_COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_ztrsm_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex       *B, magma_int_t ldb);

magma_int_t
magma_ztrtri(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_ztrtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

// ------------------------------------------------------------ zun routines
// CUDA MAGMA only
magma_int_t
magma_zungbr(
    magma_vect_t vect, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zunghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t nb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunglq(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_zungqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zungqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zungqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_zungqr2(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magma_int_t *info);

magma_int_t
magma_zunmbr(
    magma_vect_t vect, magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C, magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zunmlq(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C, magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zunmrq(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C, magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zunmql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C, magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunmql2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    const magmaDoubleComplex *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_zunmqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C, magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zunmqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex const   *tau,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magmaDoubleComplex       *hwork, magma_int_t lwork,
    magmaDoubleComplex_ptr       dT, magma_int_t nb,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunmqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    const magmaDoubleComplex *wA, magma_int_t ldwa,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunmqr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C,    magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_zunmtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C,    magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunmtr_gpu(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    const magmaDoubleComplex *wA, magma_int_t ldwa,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zunmtr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C,    magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info);

// =============================================================================
// MAGMA utility function definitions

extern const magmaDoubleComplex MAGMA_Z_NAN;
extern const magmaDoubleComplex MAGMA_Z_INF;

int magma_z_isnan( magmaDoubleComplex x );
int magma_z_isinf( magmaDoubleComplex x );
int magma_z_isnan_inf( magmaDoubleComplex x );

magmaDoubleComplex
magma_zmake_lwork( magma_int_t lwork );

magma_int_t
magma_znan_inf(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf);

magma_int_t
magma_znan_inf_gpu(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf,
    magma_queue_t queue);

void magma_zprint(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *A, magma_int_t lda);

void magma_zprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magma_queue_t queue);

void magma_zpanel_to_q(
    magma_uplo_t uplo, magma_int_t ib,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *work);

void magma_zq_to_panel(
    magma_uplo_t uplo, magma_int_t ib,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *work);

#ifdef __cplusplus
}
#endif

#undef MAGMA_COMPLEX

#endif /* MAGMA_Z_H */
