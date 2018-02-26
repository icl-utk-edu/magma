/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_VBATCHED_H
#define MAGMA_VBATCHED_H

#include "magma_types.h"

// =============================================================================
// MAGMA VBATCHED functions

#include "magma_zvbatched.h"
#include "magma_cvbatched.h"
#include "magma_dvbatched.h"
#include "magma_svbatched.h"

#ifdef __cplusplus
extern "C" {
#endif

// checker routines - LAPACK
magma_int_t 
magma_potrf_vbatched_checker(
        magma_uplo_t uplo, 
        magma_int_t* n, magma_int_t* ldda,  
        magma_int_t batchCount, magma_queue_t queue );

// checker routines - Level 3 BLAS
magma_int_t 
magma_gemm_vbatched_checker(
        magma_trans_t transA, magma_trans_t transB, 
        magma_int_t* m, magma_int_t* n, magma_int_t* k, 
        magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc,  
        magma_int_t batchCount, magma_queue_t queue );

magma_int_t 
magma_trsm_vbatched_checker( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n,
        magma_int_t* ldda, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_syrk_vbatched_checker(
        magma_int_t icomplex, 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_herk_vbatched_checker( 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_syr2k_vbatched_checker(
        magma_int_t icomplex, 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_her2k_vbatched_checker( 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_trmm_vbatched_checker(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_hemm_vbatched_checker(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc,  
        magma_int_t batchCount, magma_queue_t queue );
        
// checker routines - Level 2 BLAS
magma_int_t 
magma_gemv_vbatched_checker(
        magma_trans_t trans, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy,  
        magma_int_t batchCount, magma_queue_t queue );

magma_int_t 
magma_hemv_vbatched_checker(
        magma_uplo_t uplo, 
        magma_int_t* n, magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy,  
        magma_int_t batchCount, magma_queue_t queue );

// checker routines - Level 1 BLAS
magma_int_t 
magma_axpy_vbatched_checker( 
        magma_int_t *n, 
        magma_int_t *incx, magma_int_t *incy, 
        magma_int_t batchCount, magma_queue_t queue);

// routines to find the maximum dimensions
void magma_imax_size_1(magma_int_t *n, magma_int_t l, magma_queue_t queue);

void magma_imax_size_2(magma_int_t *m, magma_int_t *n, magma_int_t l, magma_queue_t queue);

void magma_imax_size_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, magma_int_t l, magma_queue_t queue);

// aux. routines 
magma_int_t 
magma_ivec_max( magma_int_t vecsize, 
                  magma_int_t* x, 
                  magma_int_t* work, magma_int_t lwork, magma_queue_t queue);

 
magma_int_t 
magma_isum_reduce( magma_int_t vecsize, 
                   magma_int_t* x, 
                   magma_int_t* work, magma_int_t lwork, magma_queue_t queue);

void 
magma_ivec_add( magma_int_t vecsize, 
                      magma_int_t a1, magma_int_t *x1, 
                      magma_int_t a2, magma_int_t *x2, 
                      magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_mul( magma_int_t vecsize, 
                      magma_int_t *x1, magma_int_t *x2, 
                      magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_ceildiv( magma_int_t vecsize, 
                   magma_int_t *x, 
                   magma_int_t nb, 
                   magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_roundup( magma_int_t vecsize, 
                   magma_int_t *x, 
                   magma_int_t nb, 
                   magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_setc( magma_int_t vecsize, 
                           magma_int_t *x, 
                           magma_int_t value, 
                           magma_queue_t queue);

void 
magma_zsetvector_const( magma_int_t vecsize, 
                           magmaDoubleComplex *x, 
                           magmaDoubleComplex value, 
                           magma_queue_t queue);

void 
magma_csetvector_const( magma_int_t vecsize, 
                           magmaFloatComplex *x, 
                           magmaFloatComplex value, 
                           magma_queue_t queue);

void 
magma_dsetvector_const( magma_int_t vecsize, 
                           double *x, 
                           double value, 
                           magma_queue_t queue);

void 
magma_ssetvector_const( magma_int_t vecsize, 
                           float *x, 
                           float value, 
                           magma_queue_t queue);

void 
magma_ivec_addc( magma_int_t vecsize, 
                     magma_int_t *x, magma_int_t value, 
                     magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_mulc( magma_int_t vecsize, 
                     magma_int_t *x, magma_int_t value, 
                     magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_minc( magma_int_t vecsize, 
                     magma_int_t *x, magma_int_t value, 
                     magma_int_t *y, magma_queue_t queue);

void 
magma_ivec_maxc( magma_int_t vecsize, 
                     magma_int_t* x, magma_int_t value, 
                     magma_int_t* y, magma_queue_t queue);

void 
magma_compute_trsm_jb(
    magma_int_t vecsize, magma_int_t* m, 
    magma_int_t tri_nb, magma_int_t* jbv, 
    magma_queue_t queue);

void 
magma_prefix_sum_inplace(magma_int_t* ivec, magma_int_t length, magma_queue_t queue);

void 
magma_prefix_sum_outofplace(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_queue_t queue);

void 
magma_prefix_sum_inplace_w(magma_int_t* ivec, magma_int_t length, magma_int_t* workspace, magma_int_t lwork, magma_queue_t queue);

void 
magma_prefix_sum_outofplace_w(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_int_t* workspace, magma_int_t lwork, magma_queue_t queue);

void 
magma_imax_size_1(magma_int_t *n, magma_int_t l, magma_queue_t queue);

void 
magma_imax_size_2(magma_int_t *m, magma_int_t *n, magma_int_t l, magma_queue_t queue);

void 
magma_imax_size_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, magma_int_t l, magma_queue_t queue);

#ifdef __cplusplus
}
#endif


#endif /* MAGMA_VBATCHED_H */
