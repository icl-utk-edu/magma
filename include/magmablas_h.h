/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

*/

#ifndef MAGMABLAS_H_H
#define MAGMABLAS_H_H

#include "magma_types.h"
#include "magma_copy.h"

// Half precision routines are available for C++ compilers only
#ifdef __cplusplus
extern "C" {

// =============================================================================
// conversion routines
void
magmablas_slag2h(
    magma_int_t m, magma_int_t n,
    float const * dA, magma_int_t lda,
    magmaHalf* dHA, magma_int_t ldha,
    magma_int_t *info, magma_queue_t queue);

void
magmablas_hlag2s(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA, magma_int_t lda,
    float             *dSA, magma_int_t ldsa,
    magma_queue_t queue );

void
magmablas_slag2h_batched(
    magma_int_t m, magma_int_t n,
    float const * const * dAarray, magma_int_t lda,
    magmaHalf** dHAarray, magma_int_t ldha,
    magma_int_t *info_array, magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_hlag2s_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf const * const * dAarray, magma_int_t lda,
    float               **dSAarray, magma_int_t ldsa,
    magma_int_t batchCount, magma_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)
void
magma_hgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf_const_ptr dA, magma_int_t ldda,
    magmaHalf_const_ptr dB, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

}
#endif


#endif // MAGMABLAS_H_H
