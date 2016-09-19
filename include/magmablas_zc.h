/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
*/

#ifndef MAGMABLAS_ZC_H
#define MAGMABLAS_ZC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void
magmablas_zcaxpycp(
    magma_int_t m,
    magmaFloatComplex_ptr        r,
    magmaDoubleComplex_ptr       x,
    magmaDoubleComplex_const_ptr b,
    magmaDoubleComplex_ptr       w,
    magma_queue_t queue );

void
magmablas_zclaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr SA, magma_int_t ldsa,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue );

void
magmablas_zlag2c(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr A,  magma_int_t lda,
    magmaFloatComplex_ptr       SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_clag2z(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr SA, magma_int_t ldsa,
    magmaDoubleComplex_ptr       A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_zlat2c(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr A,  magma_int_t lda,
    magmaFloatComplex_ptr       SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_clat2z(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_const_ptr SA, magma_int_t ldsa,
    magmaDoubleComplex_ptr       A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif // MAGMABLAS_ZC_H
