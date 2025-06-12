/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#ifndef MAGMA_ZVBATCHED_H
#define MAGMA_ZVBATCHED_H

#include "magma_types.h"

#define MAGMA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   *  control and tuning
   */
void magma_get_zgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb);


  /*
   *  LAPACK vbatched routines
   */

magma_int_t
magma_zgetf2_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_zgetf2_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t ipiv_i,
    magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgetrf_vbatched(
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_vbatched_max_nocheck(
        magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magma_int_t nb, magma_int_t recnb,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t** pivinfo_array,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_vbatched_max_nocheck_work(
        magma_int_t* m, magma_int_t* n,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        void* work, magma_int_t* lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_izamax_vbatched(
        magma_int_t length, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zswap_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t magma_zscal_zgeru_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t *M, magma_int_t *N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetf2_vbatched(
    magma_int_t *m, magma_int_t *n, magma_int_t *minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);

void
magma_zlaswp_left_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_zlaswp_right_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_zlaswp_left_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_zlaswp_right_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t magma_zscal_zgeru_nopiv_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t *M, magma_int_t *N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    double *dtol_array, double eps, magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetf2_nopiv_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps, magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetf2_nopiv_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps, magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_zgetf2_nopiv_vbatched(
    magma_int_t *m, magma_int_t *n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    double *dtol_array, double eps, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_nopiv_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);
    
magma_int_t
magma_zgetrf_nopiv_vbatched_max_nocheck(
        magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magma_int_t nb, magma_int_t recnb,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        double* dtol_array, double eps,
        magma_int_t *info_array, magma_int_t batchCount,
        magma_queue_t queue);

magma_int_t
magma_zgetrf_nopiv_vbatched_max_nocheck_work(
        magma_int_t* m, magma_int_t* n,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        double *dtol_array, double eps, magma_int_t *info_array,
        void* work, magma_int_t* lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_nopiv_vbatched(
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_nopiv_expert_vbatched(
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        double *dtol_array, double eps, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_lpout_vbatched(
    magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,
    magmaDoubleComplex **dA_array, magma_int_t *lda, magma_int_t gbstep,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n,
    magmaDoubleComplex **dA_array, magma_int_t* lda,
    magmaDoubleComplex **dA_displ,
    magmaDoubleComplex **dW_displ,
    magmaDoubleComplex **dB_displ,
    magmaDoubleComplex **dC_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n,
    magma_int_t *ibvec, magma_int_t nb,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dX_array,    magma_int_t* dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ,
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n,
    magmaDoubleComplex **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_int_t max_n, magma_queue_t queue);

magma_int_t
magma_zpotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n,
    magmaDoubleComplex **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void
magmablas_zgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zherk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zherk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zherk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        double beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc,
        magma_int_t batchCount,
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zherk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        double beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zherk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        double beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zsyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc,
        magma_int_t batchCount,
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zsyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zher2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zher2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zher2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zher2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zsyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_zsyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_small_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ztrsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ztrsm_inv_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag,
    magma_int_t *m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magmaDoubleComplex** dX_array,    magma_int_t* lddx,
    magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ,
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void magmablas_ztrsm_inv_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magmaDoubleComplex** dX_array,    magma_int_t* lddx,
    magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ,
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void magmablas_ztrsm_inv_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void
magmablas_ztrsm_inv_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void
magmablas_ztrsm_inv_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magmablas_ztrsm_inv_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t* ldda,
    magmaDoubleComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magmablas_ztrtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    magmaDoubleComplex const * const *dA_array, magma_int_t *ldda,
    magmaDoubleComplex **dinvA_array,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zhemm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t *lddc,
        magma_int_t max_m, magma_int_t max_n,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
        magma_int_t specM, magma_int_t specN,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
        magma_queue_t queue );

void
magmablas_zhemm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
        magma_queue_t queue );

void
magmablas_zhemm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemm_vbatched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t *ldda,
        magmaDoubleComplex **dB_array, magma_int_t *lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_queue_t queue );

/* Level 2 */
void
magmablas_zgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_zgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_zgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zhemv_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t* n, magmaDoubleComplex alpha,
    magmaDoubleComplex **dA_array, magma_int_t* ldda,
    magmaDoubleComplex **dX_array, magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dY_array, magma_int_t* incy,
    magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_n, magma_queue_t queue);

void
magmablas_zhemv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zhemv_vbatched(
    magma_uplo_t uplo, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t* incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void magma_zset_pointer_var_cc(
    magmaDoubleComplex **output_array,
    magmaDoubleComplex *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column,
    magma_int_t *batch_offset,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magma_zdisplace_pointers_var_cc(magmaDoubleComplex **output_array,
    magmaDoubleComplex **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zdisplace_pointers_var_cv(magmaDoubleComplex **output_array,
    magmaDoubleComplex **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t* column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zdisplace_pointers_var_vc(magmaDoubleComplex **output_array,
    magmaDoubleComplex **input_array, magma_int_t* lda,
    magma_int_t *row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zdisplace_pointers_var_vv(magmaDoubleComplex **output_array,
    magmaDoubleComplex **input_array, magma_int_t* lda,
    magma_int_t* row, magma_int_t* column,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_zlaset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zlacpy_vbatched(
    magma_uplo_t uplo,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex const * const * dAarray, magma_int_t* ldda,
    magmaDoubleComplex**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  Aux. vbatched routines
   */
magma_int_t magma_get_zpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef MAGMA_COMPLEX

#endif  /* MAGMA_ZVBATCHED_H */
