/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_z
#include "trsm_template_kernel_vbatched.dp.hpp"
#include "./trsm_config/ztrsm_param.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_ztrsm_small_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nrowA = (side == MagmaLeft ? max_m : max_n);

    if( side == MagmaLeft ){
        if     ( nrowA <=  2 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB2>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <=  4 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB4>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <=  8 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB8>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <= 16 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB16>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <= 32 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_LEFT_NB32>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else
            printf("error in function %s: nrowA must be less than 32\n", __func__);
    }else{
        if     ( nrowA <=  2 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB2>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <=  4 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB4>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <=  8 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB8>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <= 16 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB16>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else if( nrowA <= 32 )
            trsm_small_vbatched<magmaDoubleComplex, ZTRSM_BATCHED_RIGHT_NB32>(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue );
        else
            printf("error in function %s: nrowA must be less than 32\n", __func__);
    }
}

