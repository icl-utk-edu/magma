/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define PRECISION_z
//-------------------------

magma_int_t magma_get_ztrsm_vbatched_nb(magma_int_t max_n)
{
    if      ( max_n > 2048 ) return 2048;
    else if ( max_n > 1024 ) return 1024;
    else if ( max_n >  512 ) return 512;
    else if ( max_n >  256 ) return 256;
    else if ( max_n >  128 ) return 128;
    else if ( max_n >   64 ) return  64;
    else if ( max_n >   32 ) return  32;
    else if ( max_n >   16 ) return  16;
    else if ( max_n >    8 ) return   8;
    else if ( max_n >    4 ) return   4;
    else if ( max_n >    2 ) return   2;
    else return 1;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_ztrsm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
#define dA_array(i,j) dA_array, i, j
#define dB_array(i,j) dB_array, i, j

    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    /*
    DPCT1064:21: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    const magmaDoubleComplex c_negone = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_neg_ialpha = MAGMA_Z_DIV(c_negone, alpha);

    if(max_m == 0 || max_n == 0) return;

    magma_int_t max_nrowA = (side == MagmaLeft ? max_m : max_n);
    magma_int_t shape = 0;
    if      (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // lNL
    else if (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // lNU
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // lTL | lCL
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // lTU | lCU
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 4; } // rNL
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 5; } // rNU
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 6; } // rTL | rCL
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 7; } // rTU | rCU

    // use the fixed-size batched function to get nb
    magma_int_t batrsm_stop_nb = magma_get_ztrsm_batched_stop_nb(side, max_m, max_n);
    // stopping condition
    if(max_nrowA <= batrsm_stop_nb){
        magmablas_ztrsm_small_vbatched(
                side, uplo, transA, diag,
                max_m, max_n, m, n, alpha,
                dA_array(Ai, Aj), ldda,
                dB_array(Bi, Bj), lddb,
                batchCount, queue );
        return;
    }

    switch(shape)
    {
        case 0: // lNl
            {
                const int m2 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m1 = max_m - m2;

                //printf("trsm1: m1 = %d, m2 = %d\n", m1, m2);
                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        m2, max_n, m1,
                        m, n, m,
                        c_negone, dA_array, Ai+m1, Aj, ldda,
                                  dB_array, Bi,    Bj, lddb,
                        alpha   , dB_array, Bi+m1, Bj, lddb,
                        batchCount, queue );

                //printf("trsm2: m1 = %d, m2 = %d\n", m1, m2);
                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, c_one,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj),    lddb,
                        batchCount, queue );
            }
            break;
        case 1: // lNU
            {
                const int m1 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m2 = max_m - m1;

                //printf("trsm1: m1 = %d, m2 = %d\n", m1, m2);
                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj),    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        m1, max_n, m2,
                        m, n, m,
                        c_neg_ialpha, dA_array, Ai, Aj+m1, ldda,
                                      dB_array, Bi+m1, Bj, lddb,
                        c_one       , dB_array, Bi,    Bj, lddb,
                        batchCount, queue );

                //printf("trsm2: m1 = %d, m2 = %d\n", m1, m2);
                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );

            }
            break;
        case 2: // lTL || lCL
            {
                const int m2 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m1 = max_m - m2;

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj), lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        transA, MagmaNoTrans,
                        m1, max_n, m2,
                        m, n, m,
                        c_neg_ialpha, dA_array, Ai+m1, Aj, ldda,
                                      dB_array, Bi+m1, Bj, lddb,
                        c_one,        dB_array, Bi,    Bj, lddb,
                        batchCount, queue );

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );
            }
            break;
        case 3: // lTU | lCU
            {
                const int m1 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m2 = max_m - m1;

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        transA, MagmaNoTrans,
                        m2, max_n, m1,
                        m, n, m,
                        c_negone, dA_array, Ai,    Aj+m1, ldda,
                                  dB_array, Bi,    Bj,    lddb,
                        alpha   , dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, c_one,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj),    lddb,
                        batchCount, queue );
            }
            break;
        case 4: // rNL
             {
                const int n2 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n1 = max_n - n2;

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n1, n2,
                        m, n, n,
                        c_neg_ialpha, dB_array, Bi,    Bj+n1, lddb,
                                      dA_array, Ai+n1, Aj,    ldda,
                        c_one       , dB_array, Bi,    Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );
            }
            break;
        case 5: // rNU
            {
                const int n1 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n2 = max_n - n1;

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n2, n1,
                        m, n, n,
                        c_negone, dB_array, Bi, Bj,    lddb,
                                  dA_array, Ai, Aj+n1, ldda,
                        alpha   , dB_array, Bi, Bj+n1, lddb,
                        batchCount, queue );

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, c_one,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue );
            }
            break;
        case 6: // rTL | rCL
            {
                const int n1 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n2 = max_n - n1;

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n2, n1,
                        m, n, n,
                        c_negone, dB_array, Bi,    Bj,    lddb,
                                  dA_array, Ai+n1, Aj,    ldda,
                        alpha   , dB_array, Bi,    Bj+n1, lddb,
                        batchCount, queue );

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, c_one,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue );
            }
            break;
        case 7: // rTU | rCU
            {
                const int n2 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n1 = max_n - n2;

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n1, n2,
                        m, n, n,
                        c_neg_ialpha, dB_array, Bi, Bj+n1, lddb,
                                      dA_array, Ai, Aj+n1, ldda,
                        c_one       , dB_array, Bi, Bj,    lddb,
                        batchCount, queue );

                magmablas_ztrsm_vbatched_core(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue );
            }
            break;
        default:; // propose something
    }
#undef dA_array
#undef dB_array
}
///////////////////////////////////////////////////////////////////////////////////////////////////
