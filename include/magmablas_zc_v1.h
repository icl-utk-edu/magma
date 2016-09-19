/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
*/

#ifndef MAGMABLAS_ZC_V1_H
#define MAGMABLAS_ZC_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void
magmablas_zcaxpycp_v1(
    magma_int_t m,
    magmaFloatComplex_ptr  r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b,
    magmaDoubleComplex_ptr w );

void
magmablas_zclaswp_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr  A, magma_int_t lda,
    magmaFloatComplex_ptr  SA,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx );

void
magmablas_zlag2c_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr  A, magma_int_t lda,
    magmaFloatComplex_ptr        SA, magma_int_t ldsa,
    magma_int_t *info );

void
magmablas_clag2z_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr  SA, magma_int_t ldsa,
    magmaDoubleComplex_ptr        A, magma_int_t lda,
    magma_int_t *info );

void
magmablas_zlat2c_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr  A, magma_int_t lda,
    magmaFloatComplex_ptr        SA, magma_int_t ldsa,
    magma_int_t *info );

void
magmablas_clat2z_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_const_ptr  SA, magma_int_t ldsa,
    magmaDoubleComplex_ptr        A, magma_int_t lda,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif // MAGMABLAS_ZC_V1_H
