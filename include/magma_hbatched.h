/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_HBATCHED_H
#define MAGMA_HBATCHED_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void magma_hset_pointer(
    magmaHalf **output_array,
    magmaHalf *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue);

magma_int_t 
magmablas_hgemm_batched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    magmaHalf alpha,
    magmaHalf const * const * dAarray, magma_int_t ldda,
    magmaHalf const * const * dBarray, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dCarray, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

#ifdef __cplusplus
}
#endif


#endif /* MAGMA_HBATCHED_H */
