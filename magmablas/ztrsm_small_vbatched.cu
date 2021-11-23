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
#include "trsm_template_kernel_vbatched.cuh"
#include "./trsm_config/ztrsm_param.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_ztrsm_small_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t* lddb, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, 
        magma_int_t spec_m, magma_int_t spec_n, 
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nrowA = (side == MagmaLeft ? max_m : max_n);

    if( side == MagmaLeft ){
        if     ( nrowA <=  2 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB2>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <=  4 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB4>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <=  8 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB8>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <= 16 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB16>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <= 32 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB32>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else
            printf("error in function %s: nrowA must be less than 32\n", __func__);
    }else{
        if     ( nrowA <=  2 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB2>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <=  4 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB4>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <=  8 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB8>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <= 16 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB16>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else if( nrowA <= 32 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB32>(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        else
            printf("error in function %s: nrowA must be less than 32\n", __func__);
    }
}

