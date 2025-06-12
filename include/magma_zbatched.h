/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/

#ifndef MAGMA_ZBATCHED_H
#define MAGMA_ZBATCHED_H

#include "magma_types.h"

#define MAGMA_COMPLEX

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void
magma_zset_pointer(
    magmaDoubleComplex **output_array,
    magmaDoubleComplex *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batch_offset,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zdisplace_pointers(
    magmaDoubleComplex **output_array,
    magmaDoubleComplex **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zrecommend_cublas_gemm_batched(
    magma_trans_t transa, magma_trans_t transb,
    magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t
magma_zrecommend_cublas_gemm_stream(
    magma_trans_t transa, magma_trans_t transb,
    magma_int_t m, magma_int_t n, magma_int_t k);

void magma_get_zpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);

magma_int_t magma_get_zpotrf_batched_crossover();

void magma_get_zgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);
magma_int_t magma_get_zgetrf_batched_ntcol(magma_int_t m, magma_int_t n);
magma_int_t magma_get_zgemm_batched_ntcol(magma_int_t n);
magma_int_t magma_get_zgemm_batched_smallsq_limit(magma_int_t n);
magma_int_t magma_get_zgeqrf_batched_nb(magma_int_t m);
magma_int_t magma_use_zgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount);
magma_int_t magma_get_zgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n);
magma_int_t magma_get_zgeqrf_batched_ntcol(magma_int_t m, magma_int_t n);
magma_int_t magma_get_zgetri_batched_ntcol(magma_int_t m, magma_int_t n);
magma_int_t magma_get_ztrsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n);
void magma_get_zgbtrf_batched_params(magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t *nb, magma_int_t *threads);

void
magmablas_zswapdblk_batched(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex **dA, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex **dB, magma_int_t lddb, magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */
void
magmablas_zgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_zgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_zgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_batched_strided(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * dA, magma_int_t ldda, magma_int_t strideA,
    magmaDoubleComplex const * dB, magma_int_t lddb, magma_int_t strideB,
    magmaDoubleComplex beta,
    magmaDoubleComplex       * dC, magma_int_t lddc, magma_int_t strideC,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyrk_batched_core(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zherk_batched_core(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zher2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    double beta, magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zsyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex const * const *dA_array, magma_int_t ldda,
    magmaDoubleComplex **dinvA_array,
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ztrsm_small_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_recursive_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrsm_inv_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ztrsm_inv_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magmaDoubleComplex** dX_array,    magma_int_t lddx,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ,
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ztrsm_inv_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magmaDoubleComplex** dX_array,    magma_int_t lddx,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ,
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ztrsv_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ztrsv_recursive_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dx_array, magma_int_t xi, magma_int_t incx,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_batched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ztrmm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemm_batched_core(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t lddc,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemm_batched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dC_array, magma_int_t lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemv_batched_core(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dX_array, magma_int_t incx,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dY_array, magma_int_t incy,
        magma_int_t offA, magma_int_t offX, magma_int_t offY,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zhemv_batched(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dX_array, magma_int_t incx,
        magmaDoubleComplex beta,
        magmaDoubleComplex **dY_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_expert_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t recnb,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotf2_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrs_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zposv_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrs_batched(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zlaswp_rowparallel_batched(
    magma_int_t n,
    magmaDoubleComplex**  input_array, magma_int_t  input_i, magma_int_t  input_j, magma_int_t ldi,
    magmaDoubleComplex** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **pivinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zlaswp_rowserial_batched(
    magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zlaswp_columnserial_batched(
    magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ztranspose_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array,  magma_int_t ldda,
    magmaDoubleComplex **dAT_array, magma_int_t lddat,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_zlaset_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zlaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbsv_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbsv_batched_fused_sm(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** ipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgbsv_batched_strided_work(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t* dipiv, magma_int_t stride_piv,
    magmaDoubleComplex* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbsv_batched_strided(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t* dipiv, magma_int_t stride_piv,
    magmaDoubleComplex* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbsv_batched_work(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zgbtrf_set_fillin(
    magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** dipiv_array, int* ju_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtf2_zswap_batched(
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    magma_int_t** dipiv_array, magma_int_t ipiv_offset,
    int* ju_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtf2_scal_ger_batched(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    int* ju_array, magma_int_t gbstep, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrf_batched_fused_sm(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgbtrf_batched_sliding_window_loopout(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgbtrf_batched_sliding_window_loopin(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgbtrf_batched_strided(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex* dAB, magma_int_t lddab, magma_int_t strideAB,
        magma_int_t* dipiv, magma_int_t stride_piv,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrf_batched_strided_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex* dAB, magma_int_t lddab, magma_int_t strideAB,
        magma_int_t* dipiv, magma_int_t stride_piv,
        magma_int_t *info_array,
        void* device_work, magma_int_t *lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrf_batched_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex **dAB_array, magma_int_t lddab,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        void* device_work, magma_int_t *lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrf_batched(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t lddab,
    magma_int_t **dipiv_array, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrs_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrs_lower_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_zgbtrs_lower_blocked_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgbtrs_swap_batched(
    magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t j,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgbtrs_upper_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_zgbtrs_upper_blocked_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgbtrs_upper_columnwise_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku,
    magma_int_t nrhs, magma_int_t j,
    magmaDoubleComplex** dA_array, magma_int_t ldda,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgeru_batched_core(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dX_array, magma_int_t xi, magma_int_t xj, magma_int_t lddx, magma_int_t incx,
    magmaDoubleComplex** dY_array, magma_int_t yi, magma_int_t yj, magma_int_t lddy, magma_int_t incy,
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgesv_batched_small(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgetf2_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t **dpivinfo_array,
    magma_int_t *info_array,
    magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_recpanel_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);

magma_int_t
magma_zgetrf_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array,
    magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetf2_fused_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_zgetrf_batched_smallsq_noshfl(
    magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgetri_outofplace_batched(
    magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magmaDoubleComplex **dinvA_array, magma_int_t lddia,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zdisplace_intpointers(
    magma_int_t **output_array,
    magma_int_t **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_izamax_atomic_batched(
    magma_int_t n,
    magmaDoubleComplex** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void
magmablas_izamax_tree_batched(
    magma_int_t n,
    magmaDoubleComplex** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void
magmablas_izamax_batched(
    magma_int_t n,
    magmaDoubleComplex** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void
magmablas_izamax(
    magma_int_t n,
    magmaDoubleComplex* x, magma_int_t incx,
    magma_int_t *max_id);

magma_int_t
magma_izamax_batched(
        magma_int_t length,
        magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t lda, magma_int_t incx,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t step, magma_int_t gbstep, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zswap_batched(
    magma_int_t n, magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx,
    magma_int_t step, magma_int_t** ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zscal_zgeru_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zcomputecolumn_batched(
    magma_int_t m, magma_int_t paneloffset, magma_int_t step,
    magmaDoubleComplex **dA_array,  magma_int_t lda,
    magma_int_t ai, magma_int_t aj,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zgetf2trsm_batched(
    magma_int_t ib, magma_int_t n,
    magmaDoubleComplex **dA_array,  magma_int_t j, magma_int_t lda,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetf2_nopiv_internal_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_recpanel_nopiv_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW1_displ, magmaDoubleComplex** dW2_displ,
    magmaDoubleComplex** dW3_displ, magmaDoubleComplex** dW4_displ,
    magmaDoubleComplex** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrf_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array,
    magma_int_t lda,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgetrs_nopiv_batched(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgesv_nopiv_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgesv_rbt_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgesv_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magmaDoubleComplex *U, magmaDoubleComplex *V,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zprbt_batched(
    magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex *du, magmaDoubleComplex *dv,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zprbt_mv_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *dv, magmaDoubleComplex **db_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zprbt_mtv_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *du, magmaDoubleComplex **db_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
setup_pivinfo(
    magma_int_t *pivinfo, magma_int_t *ipiv,
    magma_int_t m, magma_int_t nb,
    magma_queue_t queue);

void
magmablas_zgeadd_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zlacpy_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex_ptr             dBarray[], magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zgemv_batched_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dx_array, magma_int_t xi, magma_int_t incx,
    const magmaDoubleComplex beta,
    magmaDoubleComplex** dy_array, magma_int_t yi, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zgemv_batched_internal(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, const magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    magmaDoubleComplex const * const * dx_array, const magmaDoubleComplex* dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    const magmaDoubleComplex beta,
    magmaDoubleComplex** dy_array, magmaDoubleComplex* dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zgemv_batched_strided_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t Ai, magma_int_t Aj, magma_int_t strideA,
    const magmaDoubleComplex* dx, magma_int_t incx, magma_int_t xi, magma_int_t stridex,
    const magmaDoubleComplex beta,
          magmaDoubleComplex* dy, magma_int_t incy, magma_int_t yi, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dx_array, magma_int_t incx,
    const magmaDoubleComplex beta,
    magmaDoubleComplex** dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_zgemv_batched_strided(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    const magmaDoubleComplex* dx, magma_int_t incx, magma_int_t stridex,
    const magmaDoubleComplex beta,
    magmaDoubleComplex* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_zgemv_batched_smallsq(
    magma_trans_t trans, magma_int_t n,
    const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dx_array, magma_int_t xi, magma_int_t incx,
    const magmaDoubleComplex beta,
    magmaDoubleComplex** dy_array, magma_int_t yi, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_zgemv_batched_strided_smallsq(
    magma_trans_t transA, magma_int_t n,
    const magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magma_int_t strideA,
    const magmaDoubleComplex* dx, magma_int_t xi, magma_int_t incx, magma_int_t stridex,
    const magmaDoubleComplex beta,
    magmaDoubleComplex* dy, magma_int_t yi, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqrf_batched_smallsq(
    magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgeqrf_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array,
    magma_int_t lda,
    magmaDoubleComplex **dtau_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqrf_batched_work(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dtau_array,
    magma_int_t *info_array,
    void* device_work, magma_int_t* device_lwork,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqrf_expert_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dR_array, magma_int_t lddr,
    magmaDoubleComplex **dT_array, magma_int_t lddt,
    magmaDoubleComplex **dtau_array, magma_int_t provide_RT,
    magmaDoubleComplex **dW_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqrf_batched_v4(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array,
    magma_int_t lda,
    magmaDoubleComplex **tau_array,
    magma_int_t *info_array,
    magma_int_t batchCount);

magma_int_t
magma_zgeqrf_panel_fused_update_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex** tau_array, magma_int_t taui,
        magmaDoubleComplex** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr,
        magma_int_t *info_array, magma_int_t separate_R_V,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqrf_panel_internal_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex** tau_array, magma_int_t taui,
        magmaDoubleComplex** dT_array, magma_int_t Ti, magma_int_t Tj, magma_int_t lddt,
        magmaDoubleComplex** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr,
        magmaDoubleComplex** dwork_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqrf_panel_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** tau_array,
    magmaDoubleComplex** dT_array, magma_int_t ldt,
    magmaDoubleComplex** dR_array, magma_int_t ldr,
    magmaDoubleComplex** dwork_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgels_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zgeqr2_fused_reg_tall_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgeqr2_fused_reg_medium_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgeqr2_fused_reg_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgeqr2_fused_sm_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zgeqr2_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zlarf_fused_reg_tall_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zlarf_fused_reg_medium_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zlarf_fused_reg_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zlarf_fused_sm_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_zlarfb_gemm_internal_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV_array[],    magma_int_t vi, magma_int_t vj, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT_array[],    magma_int_t Ti, magma_int_t Tj, magma_int_t lddt,
    magmaDoubleComplex_ptr dC_array[],          magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magmaDoubleComplex_ptr dwork_array[],       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zlarfb_gemm_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV_array[],    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT_array[],    magma_int_t lddt,
    magmaDoubleComplex_ptr dC_array[],          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork_array[],       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zlarft_internal_batched(
        magma_int_t n, magma_int_t k, magma_int_t stair_T,
        magmaDoubleComplex **v_array,   magma_int_t vi, magma_int_t vj, magma_int_t ldv,
        magmaDoubleComplex **tau_array, magma_int_t taui,
        magmaDoubleComplex **T_array,   magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
        magmaDoubleComplex **work_array, magma_int_t lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zlarft_batched(
    magma_int_t n, magma_int_t k, magma_int_t stair_T,
    magmaDoubleComplex **v_array, magma_int_t ldv,
    magmaDoubleComplex **tau_array,
    magmaDoubleComplex **T_array, magma_int_t ldt,
    magmaDoubleComplex **work_array, magma_int_t lwork,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_zlarft_sm32x32_batched(
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex **v_array, magma_int_t vi, magma_int_t vj, magma_int_t ldv,
    magmaDoubleComplex **tau_array, magma_int_t taui,
    magmaDoubleComplex **T_array, magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_zlarft_recztrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *Trec, magma_int_t ldtrec,
    magmaDoubleComplex *Ttri, magma_int_t ldttri,
    magma_queue_t queue);

void magmablas_zlarft_recztrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **tau_array,  magma_int_t taui,
    magmaDoubleComplex **Trec_array, magma_int_t Treci, magma_int_t Trecj, magma_int_t ldtrec,
    magmaDoubleComplex **Ttri_array, magma_int_t Ttrii, magma_int_t Ttrij, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_zlarft_ztrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *Tin, magma_int_t ldtin,
    magmaDoubleComplex *Tout, magma_int_t ldtout,
    magma_queue_t queue);

void magmablas_zlarft_ztrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **tau_array, magma_int_t taui,
    magmaDoubleComplex **Tin_array, magma_int_t Tini, magma_int_t Tinj, magma_int_t ldtin,
    magmaDoubleComplex **Tout_array, magma_int_t Touti, magma_int_t Toutj, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dznrm2_cols_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    double **dxnorm_array,
    magma_int_t batchCount);

void
magma_zlarfgx_batched(
    magma_int_t n, magmaDoubleComplex **dx0_array, magmaDoubleComplex **dx_array,
    magmaDoubleComplex **dtau_array, double **dxnorm_array,
    magmaDoubleComplex **dR_array, magma_int_t it,
    magma_int_t batchCount);

void
magma_zlarfx_batched_v4(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **v_array,
    magmaDoubleComplex **tau_array,
    magmaDoubleComplex **C_array, magma_int_t ldc, double **xnorm_array,
    magma_int_t step,
    magma_int_t batchCount);

void
magmablas_zlarfg_batched(
    magma_int_t n,
    magmaDoubleComplex** dalpha_array,
    magmaDoubleComplex** dx_array, magma_int_t incx,
    magmaDoubleComplex** dtau_array,
    magma_int_t batchCount );

magma_int_t
magma_zpotrf_lpout_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_zpotrf_lpin_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

// host interface
void
blas_zlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex const * const * hA_array, magma_int_t lda,
    magmaDoubleComplex               **hB_array, magma_int_t ldb,
    magma_int_t batchCount );

void
blas_zgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * hA_array, magma_int_t lda,
    magmaDoubleComplex const * const * hB_array, magma_int_t ldb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **hC_array, magma_int_t ldc,
    magma_int_t batchCount );

void
blas_ztrsm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **hA_array, magma_int_t lda,
        magmaDoubleComplex **hB_array, magma_int_t ldb,
        magma_int_t batchCount );

void
blas_ztrmm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **hA_array, magma_int_t lda,
        magmaDoubleComplex **hB_array, magma_int_t ldb,
        magma_int_t batchCount );

void
blas_zhemm_batched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **hA_array, magma_int_t lda,
        magmaDoubleComplex **hB_array, magma_int_t ldb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **hC_array, magma_int_t ldc,
        magma_int_t batchCount );

void
blas_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda,
    double beta,  magmaDoubleComplex               **hC_array, magma_int_t ldc,
    magma_int_t batchCount );

void
blas_zher2k_batched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda,
                              magmaDoubleComplex const * const * hB_array, magma_int_t ldb,
    double beta,              magmaDoubleComplex               **hC_array, magma_int_t ldc,
    magma_int_t batchCount );

// for debugging purpose
void
zset_stepinit_ipiv(
    magma_int_t **ipiv_array,
    magma_int_t pm,
    magma_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef MAGMA_COMPLEX

#endif  /* MAGMA_ZBATCHED_H */
