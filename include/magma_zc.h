/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
*/

#ifndef MAGMA_ZC_H
#define MAGMA_ZC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MAGMA mixed precision function definitions
//
// In alphabetical order of base name (ignoring precision).
magma_int_t
magma_zcgeqrsv_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_zcgesv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex_ptr dworkd,
    magmaFloatComplex_ptr  dworks,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_zcgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr  dA, magma_int_t ldda,
    magmaInt_ptr        dipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaFloatComplex_ptr dSX,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_zchesv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex_ptr dworkd,
    magmaFloatComplex_ptr  dworks,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_zcposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex_ptr dworkd,
    magmaFloatComplex_ptr  dworks,
    magma_int_t *iter,
    magma_int_t *info);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_ZC_H */
