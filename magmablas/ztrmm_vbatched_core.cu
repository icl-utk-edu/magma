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
#include "batched_kernel_param.h"

#define PRECISION_z
#include "trmm_template_kernel_vbatched.cuh"

magmaDoubleComplex* magma_zptrb[2];
magmaDoubleComplex* magma_zptra[2];
magmaDoubleComplex* magma_zA;
magmaDoubleComplex* magma_zB;

magma_int_t magma_get_ztrmm_vbatched_nb(magma_int_t n)
{
    if      ( n > 2048 ) return 2048;
    else if ( n > 1024 ) return 1024;
    else if ( n >  512 ) return 512;
    else if ( n >  256 ) return 256;
    else if ( n >  128 ) return 128;
    else if ( n >   64 ) return  64;
    else if ( n >   32 ) return  32;
    else if ( n >   16 ) return  16;
    else if ( n >    8 ) return   8;
    else if ( n >    4 ) return   4;
    else if ( n >    2 ) return   2;
    else return 1;
}
/******************************************************************************/
void
magmablas_ztrmm_small_vbatched(
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
    magma_int_t shape = 0;
    if      (side == MagmaLeft  && transA == MagmaNoTrans   ) { shape = 0; } // left  - NoTrans   (lNx)
    else if (side == MagmaLeft  && transA == MagmaTrans     ) { shape = 1; } // left  - Trans     (lTx)
    else if (side == MagmaLeft  && transA == MagmaConjTrans ) { shape = 2; } // left  - ConjTrans (lCx)
    else if (side == MagmaRight && transA == MagmaNoTrans   ) { shape = 3; } // right - NoTrans   (rNx)
    else if (side == MagmaRight && transA == MagmaTrans     ) { shape = 4; } // right - Trans     (rTx)
    else if (side == MagmaRight && transA == MagmaConjTrans ) { shape = 5; } // right - ConjTrans (rCx)
    
    switch(shape)
    {
        case 0: // lNx
            trmm_template_vbatched_lNx<magmaDoubleComplex, ZTRMM_BATCHED_NB>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue);
            break;
        case 1: // lTx
            trmm_template_vbatched_lTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 0>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue);
            break;
        case 2: // lCx
            trmm_template_vbatched_lTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 1>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue);
            break;
        case 3: // rNx
            trmm_template_vbatched_rNx<magmaDoubleComplex, ZTRMM_BATCHED_NB>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue);
            break;
        case 4: // rTx
            trmm_template_vbatched_rTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 0>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue);
            break;
        case 5: // rCx
            trmm_template_vbatched_rTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 1>
            (uplo, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue);
            break;
        default:; // propose something
    }
}
/******************************************************************************/
extern "C" void 
magmablas_ztrmm_vbatched_core(
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
    const magmaDoubleComplex c_one = MAGMA_Z_ONE; 
    
    magma_int_t max_nrowA = (side == MagmaLeft ? max_m : max_n);
    // stopping condition
    if(max_nrowA <= ZTRMM_BATCHED_NB){
        magmablas_ztrmm_small_vbatched( side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, max_m, max_n, roffA, coffA, roffB, coffB, spec_m, spec_n, batchCount, queue );
        magma_queue_sync( queue );
        return;
    }
    
    magma_int_t shape = 0;
    if      (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // lNL
    else if (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // lNU
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // lTL | lCL
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // lTU | lCU
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 4; } // rNL
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 5; } // rNU
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 6; } // rTL | rCL
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 7; } // rTU | rCU
    
    // at this point we can say that max_nrowA > ZTRMM_BATCHED_NB
    switch(shape)
    {
        case 0: // lNl
            {
                const int m1 = magma_get_ztrmm_vbatched_nb(max_m); 
                const int m2 = max_m - m1;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m2, max_n, 
                        roffA+m1, coffA+m1, roffB+m1, coffB, 
                        m2, 0, 
                        batchCount, queue );
                
                magmablas_zgemm_vbatched_core( 
                        MagmaNoTrans, MagmaNoTrans, 
                        m, n, m, 
                        alpha, dA_array, ldda, 
                               dB_array, lddb, 
                        c_one, dB_array, lddb, 
                        m2, max_n, m1, 
                        roffA+m1, coffA, roffB, coffB, roffB+m1, coffB, 
                        m2, 0, m1, 
                        batchCount, queue );
                
                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m1, max_n, 
                        roffA, coffA, roffB, coffB, 
                        m1, 0, 
                        batchCount, queue );
            }
            break;
        case 1: // lNU
            {
                const int m2 = magma_get_ztrmm_vbatched_nb(max_m); 
                const int m1 = max_m - m2;
                
                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m1, max_n, 
                        roffA, coffA, roffB, coffB, 
                        m1, 0, 
                        batchCount, queue );
                        
                magmablas_zgemm_vbatched_core( 
                        MagmaNoTrans, MagmaNoTrans, 
                        m, n, m, 
                        alpha, dA_array, ldda, 
                               dB_array, lddb, 
                        c_one, dB_array, lddb, 
                        m1, max_n, m2, 
                        roffA, coffA+m1, roffB+m1, coffB, roffB, coffB, 
                        m1, 0, m2, 
                        batchCount, queue );
                        
                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m2, max_n, 
                        roffA+m1, coffA+m1, roffB+m1, coffB, 
                        m2, 0, 
                        batchCount, queue );
            }
            break;  
        case 2: // lTL || lCL
            {
                const int m2 = magma_get_ztrmm_vbatched_nb(max_m); 
                const int m1 = max_m - m2;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m1, max_n, 
                        roffA, coffA, roffB, coffB, 
                        m1, 0, 
                        batchCount, queue );

                magmablas_zgemm_vbatched_core( 
                        transA, MagmaNoTrans, 
                        m, n, m, 
                        alpha, dA_array, ldda, 
                               dB_array, lddb, 
                        c_one, dB_array, lddb, 
                        m1, max_n, m2, 
                        roffA+m1, coffA, roffB+m1, coffB, roffB, coffB, 
                        m1, 0, m2, 
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m2, max_n, 
                        roffA+m1, coffA+m1, roffB+m1, coffB, 
                        m2, 0, 
                        batchCount, queue );
            }
            break;
        case 3: // lTU | lCU
            {
                const int m1 = magma_get_ztrmm_vbatched_nb(max_m); 
                const int m2 = max_m - m1;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m2, max_n, 
                        roffA+m1, coffA+m1, roffB+m1, coffB, 
                        m2, 0, 
                        batchCount, queue );

                magmablas_zgemm_vbatched_core( 
                        transA, MagmaNoTrans, 
                        m, n, m, 
                        alpha, dA_array, ldda, 
                               dB_array, lddb, 
                        c_one, dB_array, lddb, 
                        m2, max_n, m1, 
                        roffA, coffA+m1, roffB, coffB, roffB+m1, coffB, 
                        m2, 0, m1, 
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        m1, max_n, 
                        roffA, coffA, roffB, coffB, 
                        m1, 0, 
                        batchCount, queue );
            }
            break;
        case 4: // rNL
            {
                const int n2 = magma_get_ztrmm_vbatched_nb(max_n); 
                const int n1 = max_n - n2;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n1, 
                        roffA, coffA, roffB, coffB, 
                        0, n1, 
                        batchCount, queue );

                magmablas_zgemm_vbatched_core( 
                        MagmaNoTrans, transA, 
                        m, n, n, 
                        alpha, dB_array, lddb, 
                               dA_array, ldda, 
                        c_one, dB_array, lddb, 
                        max_m, n1, n2, 
                        roffB, coffB+n1, roffA+n1, coffA, roffB, coffB, 
                        0, n1, n2, 
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n2, 
                        roffA+n1, coffA+n1, roffB, coffB+n1, 
                        0, n2, 
                        batchCount, queue );
            }
            break;
        case 5: // rNU
            {
                const int n1 = magma_get_ztrmm_vbatched_nb(max_n); 
                const int n2 = max_n - n1;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n2, 
                        roffA+n1, coffA+n1, roffB, coffB+n1, 
                        0, n2, 
                        batchCount, queue );

                magmablas_zgemm_vbatched_core( 
                        MagmaNoTrans, transA, 
                        m, n, n, 
                        alpha, dB_array, lddb, 
                               dA_array, ldda, 
                        c_one, dB_array, lddb, 
                        max_m, n2, n1, 
                        roffB, coffB, roffA, coffA+n1, roffB, coffB+n1, 
                        0, n2, n1, 
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb,
                        max_m, n1,  
                        roffA, coffA, roffB, coffB, 
                        0, n1, 
                        batchCount, queue );
            }
            break;
        case 6: // rTL | rCL
            {
                const int n1 = magma_get_ztrmm_vbatched_nb(max_n); 
                const int n2 = max_n - n1;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n2, 
                        roffA+n1, coffA+n1, roffB, coffB+n1, 
                        0, n2, 
                        batchCount, queue );

                magmablas_zgemm_vbatched_core( 
                        MagmaNoTrans, transA, 
                        m, n, n, 
                        alpha, dB_array, lddb, 
                               dA_array, ldda, 
                        c_one, dB_array, lddb, 
                        max_m, n2, n1, 
                        roffB, coffB, roffA+n1, coffA, roffB, coffB+n1, 
                        0, n2, n1, 
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n1, 
                        roffA, coffA, roffB, coffB, 
                        0, n1, 
                        batchCount, queue );
            }
            break;
        case 7: // rTU | rCU
            {
                const int n2 = magma_get_ztrmm_vbatched_nb(max_n); 
                const int n1 = max_n - n2;

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n1, 
                        roffA, coffA, roffB, coffB, 
                        0, n1, 
                        batchCount, queue );

                magmablas_zgemm_vbatched_core( 
                        MagmaNoTrans, transA, 
                        m, n, n, 
                        alpha, dB_array, lddb, 
                               dA_array, ldda, 
                        c_one, dB_array, lddb, 
                        max_m, n1, n2, 
                        roffB, coffB+n1, roffA, coffA+n1, roffB, coffB, 
                        0, n1, n2, 
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core( 
                        side, uplo, transA, diag, 
                        m, n, alpha, 
                        dA_array, ldda, 
                        dB_array, lddb, 
                        max_m, n2, 
                        roffA+n1, coffA+n1, roffB, coffB+n1, 
                        0, n2, 
                        batchCount, queue );
            }
            break;
        default:; // propose something
    }
}
/******************************************************************************/
