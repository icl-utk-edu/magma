/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Natalie Beams

*/

#ifndef MAGMA_TUNING_TREES_H
#define MAGMA_TUNING_TREES_H

#include "magma_types.h"

// =============================================================================
// Internal routines

magma_int_t
magma_zgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t
magma_cgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t
magma_dgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t
magma_sgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k);

#endif
