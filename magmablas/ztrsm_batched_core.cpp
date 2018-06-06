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

#define PRECISION_z
magma_int_t magma_get_ztrsm_batched_nb(magma_int_t n)
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

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void 
magmablas_ztrsm_recursive_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,  
        magma_int_t batchCount, magma_queue_t queue )
{
#define dA_array(i,j) dA_array, i, j
#define dB_array(i,j) dB_array, i, j

    const magmaDoubleComplex c_one    = MAGMA_Z_ONE; 
    const magmaDoubleComplex c_negone = MAGMA_Z_NEG_ONE; 
    
    magma_int_t nrowA = (side == MagmaLeft ? m : n);
    magma_int_t shape = 0;
    if      (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // lNL
    else if (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // lNU
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // lTL | lCL
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // lTU | lCU
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 4; } // rNL
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 5; } // rNU
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 6; } // rTL | rCL
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 7; } // rTU | rCU
    
    magma_int_t batrsm_stop_nb = magma_get_ztrsm_batched_stop_nb(side, m, n);
    // stopping condition
    if(nrowA <= batrsm_stop_nb){
        magmablas_ztrsm_small_batched(side, uplo, transA, diag, m, n, alpha, dA_array(Ai, Aj), ldda, dB_array(Bi, Bj), lddb, batchCount, queue );
        return;
    }
    
    switch(shape)
    {
        case 0: // lNl
            {
                const int m2 = magma_get_ztrsm_batched_nb(m); 
                const int m1 = m - m2;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m1, n, alpha, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb,  
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        MagmaNoTrans, MagmaNoTrans, 
                        m2, n, m1, 
                        c_negone, dA_array(Ai+m1, Aj), ldda, 
                                  dB_array(Bi   , Bj), lddb, 
                        alpha   , dB_array(Bi+m1, Bj), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m2, n, c_one, 
                        dA_array(Ai+m1, Aj+m1), ldda, 
                        dB_array(Bi+m1,    Bj), lddb,  
                        batchCount, queue );
            }
            break;
        case 1: // lNU
            {
                const int m2 = magma_get_ztrsm_batched_nb(m); 
                const int m1 = m - m2;
                
                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m2, n, alpha, 
                        dA_array(Ai+m1, Aj+m1), ldda, 
                        dB_array(Bi+m1,    Bj), lddb, 
                        batchCount, queue );
                        
                magma_zgemm_batched_core( 
                        MagmaNoTrans, MagmaNoTrans, 
                        m1, n, m2, 
                        c_negone, dA_array(Ai   , Aj+m1), ldda, 
                                  dB_array(Bi+m1,    Bj), lddb, 
                        alpha   , dB_array(Bi   ,    Bj), lddb, 
                        batchCount, queue );
                        
                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m1, n, c_one, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb, 
                        batchCount, queue );
            }
            break;  
        case 2: // lTL || lCL
            {
                const int m2 = magma_get_ztrsm_batched_nb(m); 
                const int m1 = m - m2;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m2, n, alpha, 
                        dA_array(Ai+m1, Aj+m1), ldda, 
                        dB_array(Bi+m1,    Bj), lddb, 
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        transA, MagmaNoTrans, 
                        m1, n, m2, 
                        c_negone, dA_array(Ai+m1, Aj), ldda, 
                                  dB_array(Bi+m1, Bj), lddb, 
                        alpha,    dB_array(Bi   , Bj), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m1, n, c_one, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb, 
                        batchCount, queue );
            }
            break;
        case 3: // lTU | lCU
            {
                const int m1 = magma_get_ztrsm_batched_nb(m); 
                const int m2 = m - m1;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m1, n, alpha, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb, 
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        transA, MagmaNoTrans, 
                        m2, n, m1, 
                        c_negone, dA_array(Ai   , Aj+m1), ldda, 
                                  dB_array(Bi   ,    Bj), lddb, 
                        alpha   , dB_array(Bi+m1,    Bj), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m2, n, c_one, 
                        dA_array(Ai+m1, Aj+m1), ldda, 
                        dB_array(Bi+m1,    Bj), lddb, 
                        batchCount, queue );
            }
            break;
        case 4: // rNL
             {
                const int n2 = magma_get_ztrsm_batched_nb(n); 
                const int n1 = n - n2;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n2, alpha, 
                        dA_array(Ai+n1, Aj+n1), ldda, 
                        dB_array(Bi,    Bj+n1), lddb, 
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        MagmaNoTrans, transA, 
                        m, n1, n2, 
                        c_negone, dB_array(Bi   , Bj+n1), lddb, 
                                  dA_array(Ai+n1,    Aj), ldda, 
                        alpha   , dB_array(Bi   ,    Bj), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n1, c_one, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb, 
                        batchCount, queue );
            }
            break;
        case 5: // rNU
            {
                const int n1 = magma_get_ztrsm_batched_nb(n); 
                const int n2 = n - n1;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n1, alpha, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb, 
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        MagmaNoTrans, transA, 
                        m, n2, n1, 
                        c_negone, dB_array(Bi,    Bj), lddb, 
                                  dA_array(Ai, Aj+n1), ldda, 
                        alpha   , dB_array(Bi, Bj+n1), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n2, c_one, 
                        dA_array(Ai+n1, Aj+n1), ldda, 
                        dB_array(Bi,    Bj+n1), lddb, 
                        batchCount, queue );
            }
            break;
        case 6: // rTL | rCL
            {
                const int n1 = magma_get_ztrsm_batched_nb(n); 
                const int n2 = n - n1;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n1, alpha, 
                        dA_array(Ai, Aj), ldda, 
                        dB_array(Bi, Bj), lddb, 
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        MagmaNoTrans, transA, 
                        m, n2, n1, 
                        c_negone, dB_array(Bi   ,    Bj), lddb, 
                                  dA_array(Ai+n1,    Aj), ldda, 
                        alpha   , dB_array(Bi   , Bj+n1), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n2, c_one, 
                        dA_array(Ai+n1, Aj+n1), ldda, 
                        dB_array(Bi,    Bj+n1), lddb, 
                        batchCount, queue );
            }
            break;
        case 7: // rTU | rCU
            {
                const int n2 = magma_get_ztrsm_batched_nb(n); 
                const int n1 = n - n2;

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n2, alpha, 
                        dA_array(Ai+n1, Aj+n1), ldda, 
                        dB_array(Bi,    Bj+n1), lddb, 
                        batchCount, queue );

                magma_zgemm_batched_core( 
                        MagmaNoTrans, transA, 
                        m, n1, n2, 
                        c_negone, dB_array(Bi, Bj+n1), lddb, 
                                  dA_array(Ai, Aj+n1), ldda, 
                        alpha   , dB_array(Bi,    Bj), lddb, 
                        batchCount, queue );

                magmablas_ztrsm_recursive_batched( 
                        side, uplo, transA, diag, 
                        m, n1, c_one, 
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
// TODO: documentation
extern "C" void 
magmablas_ztrsm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **dA_array, magma_int_t ldda,
        magmaDoubleComplex **dB_array, magma_int_t lddb, 
        magma_int_t batchCount, magma_queue_t queue )
{
#define dA_array(i,j) dA_array, i, j
#define dB_array(i,j) dB_array, i, j

    magma_int_t nrowA = (side == MagmaLeft ? m : n);
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (lddb < max(1,m)) {
        info = -11;
    } else if (batchCount < 0) {
        info = -12;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m <= 0 || n <= 0 )
        return;
    
    magmablas_ztrsm_recursive_batched( 
            side, uplo, transA, diag, 
            m, n, 
            alpha, dA_array(0,0), ldda, 
                   dB_array(0,0), lddb, 
            batchCount, queue );

#undef dA_array
#undef dB_array
}
///////////////////////////////////////////////////////////////////////////////////////////////////
