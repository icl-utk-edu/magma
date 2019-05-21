/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah
       
*/
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_z
#include "trsm_template_kernel_batched.cuh"
#include "./trsm_config/ztrsm_param.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_ztrsm_small_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, 
        magma_int_t batchCount, magma_queue_t queue )
{
#define dA_array(i,j) dA_array, i, j
#define dB_array(i,j) dB_array, i, j

    magma_int_t nrowA = (side == MagmaLeft ? m : n);

    if( side == MagmaLeft ){
        if     ( nrowA <=  2 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB2>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <=  4 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB4>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <=  8 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB8>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <= 16 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB16>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <= 32 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB32>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else
            printf("error in function %s: nrowA must be less than 32\n", __func__);
    }else{
        if     ( nrowA <=  2 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB2>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <=  4 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB4>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <=  8 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB8>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <= 16 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB16>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else if( nrowA <= 32 )
            trsm_small_batched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB32>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, Ai, Aj, Bi, Bj, batchCount, queue );
        else
            printf("error in function %s: nrowA must be less than 32\n", __func__);
    }
#undef dA_array
#undef dB_array
}

