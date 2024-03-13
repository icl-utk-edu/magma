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
#include "batched_kernel_param.h"

#define PRECISION_z
#include "trmm_template_kernel_vbatched.dp.hpp"

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
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
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
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 1: // lTx
            trmm_template_vbatched_lTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 0>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 2: // lCx
            trmm_template_vbatched_lTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 1>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 3: // rNx
            trmm_template_vbatched_rNx<magmaDoubleComplex, ZTRMM_BATCHED_NB>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 4: // rTx
            trmm_template_vbatched_rTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 0>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        case 5: // rCx
            trmm_template_vbatched_rTx<magmaDoubleComplex, ZTRMM_BATCHED_NB, 1>
            (uplo, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, max_m, max_n, batchCount, queue);
            break;
        default:; // propose something
    }
}
/******************************************************************************/
extern "C" void
magmablas_ztrmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    magma_int_t max_nrowA = (side == MagmaLeft ? max_m : max_n);
    // stopping condition
    if(max_nrowA <= ZTRMM_BATCHED_NB){
        magmablas_ztrmm_small_vbatched(
            side, uplo, transA, diag,
            max_m, max_n, m, n, alpha,
            dA_array, Ai, Aj, ldda,
            dB_array, Bi, Bj, lddb,
            batchCount, queue );
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
                        m2, max_n, m, n, alpha,
                        dA_array, Ai+m1, Aj+m1, ldda,
                        dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        m2, max_n, m1,
                        m, n, m,
                        alpha, dA_array, Ai+m1, Aj, ldda,
                               dB_array, Bi,    Bj, lddb,
                        c_one, dB_array, Bi+m1, Bj, lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );
            }
            break;
        case 1: // lNU
            {
                const int m2 = magma_get_ztrmm_vbatched_nb(max_m);
                const int m1 = max_m - m2;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        m1, max_n, m2,
                        m, n, m,
                        alpha, dA_array, Ai,    Aj+m1, ldda,
                               dB_array, Bi+m1, Bj,    lddb,
                        c_one, dB_array, Bi,    Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array, Ai+m1, Aj+m1, ldda,
                        dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );
            }
            break;
        case 2: // lTL || lCL
            {
                const int m2 = magma_get_ztrmm_vbatched_nb(max_m);
                const int m1 = max_m - m2;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        transA, MagmaNoTrans,
                        m1, max_n, m2,
                        m, n, m,
                        alpha, dA_array, Ai+m1, Aj, ldda,
                               dB_array, Bi+m1, Bj, lddb,
                        c_one, dB_array, Bi,    Bj, lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array, Ai+m1, Aj+m1, ldda,
                        dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );
            }
            break;
        case 3: // lTU | lCU
            {
                const int m1 = magma_get_ztrmm_vbatched_nb(max_m);
                const int m2 = max_m - m1;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array, Ai+m1, Aj+m1, ldda,
                        dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        transA, MagmaNoTrans,
                        m2, max_n, m1,
                        m, n, m,
                        alpha, dA_array, Ai,    Aj+m1, ldda,
                               dB_array, Bi,    Bj,    lddb,
                        c_one, dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );
            }
            break;
        case 4: // rNL
            {
                const int n2 = magma_get_ztrmm_vbatched_nb(max_n);
                const int n1 = max_n - n2;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n1, n2,
                        m, n, n,
                        alpha, dB_array, Bi,    Bj+n1, lddb,
                               dA_array, Ai+n1, Aj,    ldda,
                        c_one, dB_array, Bi,    Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array, Ai+n1, Aj+n1, ldda,
                        dB_array, Bi, Bj+n1,    lddb,
                        batchCount, queue );
            }
            break;
        case 5: // rNU
            {
                const int n1 = magma_get_ztrmm_vbatched_nb(max_n);
                const int n2 = max_n - n1;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array, Ai+n1, Aj+n1, ldda,
                        dB_array, Bi, Bj+n1,    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n2, n1,
                        m, n, n,
                        alpha, dB_array, Bi, Bj,    lddb,
                               dA_array, Ai, Aj+n1, ldda,
                        c_one, dB_array, Bi, Bj+n1, lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );
            }
            break;
        case 6: // rTL | rCL
            {
                const int n1 = magma_get_ztrmm_vbatched_nb(max_n);
                const int n2 = max_n - n1;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array, Ai+n1, Aj+n1, ldda,
                        dB_array, Bi, Bj+n1,    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n2, n1,
                        m, n, n,
                        alpha, dB_array, Bi,    Bj,    lddb,
                               dA_array, Ai+n1, Aj,    ldda,
                        c_one, dB_array, Bi,    Bj+n1, lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );
            }
            break;
        case 7: // rTU | rCU
            {
                const int n2 = magma_get_ztrmm_vbatched_nb(max_n);
                const int n1 = max_n - n2;

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array, Ai, Aj, ldda,
                        dB_array, Bi, Bj, lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n1, n2,
                        m, n, n,
                        alpha, dB_array, Bi, Bj+n1, lddb,
                               dA_array, Ai, Aj+n1, ldda,
                        c_one, dB_array, Bi, Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrmm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array, Ai+n1, Aj+n1, ldda,
                        dB_array, Bi, Bj+n1,    lddb,
                        batchCount, queue );
            }
            break;
        default:; // propose something
    }
}
/******************************************************************************/
