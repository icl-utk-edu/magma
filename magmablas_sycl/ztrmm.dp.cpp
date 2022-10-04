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
#include "batched_kernel_param.h"

#define PRECISION_z
#include "trmm_template_kernel.dp.hpp"

magma_int_t magma_get_ztrmm_nb(magma_int_t n)
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
void
magmablas_ztrmm_small(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dB, magma_int_t lddb,
        magma_queue_t queue )
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
            trmm_template_lNx<magmaDoubleComplex, ZTRMM_NB>
            (uplo, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
            break;
        case 1: // lTx
            trmm_template_lTx<magmaDoubleComplex, ZTRMM_NB, 0>
            (uplo, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
            break;
        case 2: // lCx
            trmm_template_lTx<magmaDoubleComplex, ZTRMM_NB, 1>
            (uplo, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
            break;
        case 3: // rNx
            trmm_template_rNx<magmaDoubleComplex, ZTRMM_NB>
            (uplo, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
            break;
        case 4: // rTx
            trmm_template_rTx<magmaDoubleComplex, ZTRMM_NB, 0>
            (uplo, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
            break;
        case 5: // rCx
            trmm_template_rTx<magmaDoubleComplex, ZTRMM_NB, 1>
            (uplo, diag, m, n, alpha, dA, ldda, dB, lddb, queue);
            break;
        default:; // propose something
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void
magmablas_ztrmm_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dB, magma_int_t lddb,
        magma_queue_t queue )
{
#define dA(i,j) dA + j*ldda + i
#define dB(i,j) dB + j*lddb + i

    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    magma_int_t nrowA = (side == MagmaLeft ? m : n);
    // stopping condition
    if(nrowA <= ZTRMM_NB){
        magmablas_ztrmm_small( side, uplo, transA, diag,
            m, n, alpha,
            dA, ldda,
            dB, lddb, queue );
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

    // at this point we can tell that nrowA > ZTRMM_NB
    switch(shape)
    {
        case 0: // lNl
            {
                const int m1 = magma_get_ztrmm_nb(m);
                const int m2 = m - m1;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m2, n, alpha,
                        dA(m1, m1), ldda,
                        dB(m1,  0), lddb,
                        queue );

                magma_zgemm(
                        MagmaNoTrans, MagmaNoTrans,
                        m2, n, m1,
                        alpha, dA(m1, 0), ldda,
                               dB( 0, 0), lddb,
                        c_one, dB(m1, 0), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m1, n, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );
            }
            break;
        case 1: // lNU
            {
                const int m2 = magma_get_ztrmm_nb(m);
                const int m1 = m - m2;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m1, n, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );

                magma_zgemm(
                        MagmaNoTrans, MagmaNoTrans,
                        m1, n, m2,
                        alpha, dA( 0, m1), ldda,
                               dB(m1,  0), lddb,
                        c_one, dB( 0,  0), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m2, n, alpha,
                        dA(m1, m1), ldda,
                        dB(m1,  0), lddb,
                        queue );
            }
            break;
        case 2: // lTL || lCL
            {
                const int m2 = magma_get_ztrmm_nb(m);
                const int m1 = m - m2;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m1, n, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );

                magma_zgemm(
                        transA, MagmaNoTrans,
                        m1, n, m2,
                        alpha, dA(m1, 0), ldda,
                               dB(m1, 0), lddb,
                        c_one, dB( 0, 0), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m2, n, alpha,
                        dA(m1, m1), ldda,
                        dB(m1,  0), lddb,
                        queue );
            }
            break;
        case 3: // lTU | lCU
            {
                const int m1 = magma_get_ztrmm_nb(m);
                const int m2 = m - m1;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m2, n, alpha,
                        dA(m1, m1), ldda,
                        dB(m1,  0), lddb,
                        queue );

                magma_zgemm(
                        transA, MagmaNoTrans,
                        m2, n, m1,
                        alpha, dA(0, m1), ldda,
                               dB(0,  0), lddb,
                        c_one, dB(m1, 0), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m1, n, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );
            }
            break;
        case 4: // rNL
            {
                const int n2 = magma_get_ztrmm_nb(n);
                const int n1 = n - n2;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n1, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );

                magma_zgemm(
                        MagmaNoTrans, transA,
                        m, n1, n2,
                        alpha, dB(0, n1), lddb,
                               dA(n1, 0), ldda,
                        c_one, dB( 0, 0), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n2, alpha,
                        dA(n1, n1), ldda,
                        dB( 0, n1), lddb,
                        queue );
            }
            break;
        case 5: // rNU
            {
                const int n1 = magma_get_ztrmm_nb(n);
                const int n2 = n - n1;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n2, alpha,
                        dA(n1, n1), ldda,
                        dB( 0, n1), lddb,
                        queue );

                magma_zgemm(
                        MagmaNoTrans, transA,
                        m, n2, n1,
                        alpha, dB(0,  0), lddb,
                               dA(0, n1), ldda,
                        c_one, dB(0, n1), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n1, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );
            }
            break;
        case 6: // rTL | rCL
            {
                const int n1 = magma_get_ztrmm_nb(n);
                const int n2 = n - n1;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n2, alpha,
                        dA(n1, n1), ldda,
                        dB( 0, n1), lddb,
                        queue );

                magma_zgemm(
                        MagmaNoTrans, transA,
                        m, n2, n1,
                        alpha, dB( 0,  0), lddb,
                               dA(n1,  0), ldda,
                        c_one, dB( 0, n1), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n1, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );
            }
            break;
        case 7: // rTU | rCU
            {
                const int n2 = magma_get_ztrmm_nb(n);
                const int n1 = n - n2;

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n1, alpha,
                        dA(0, 0), ldda,
                        dB(0, 0), lddb,
                        queue );

                magma_zgemm(
                        MagmaNoTrans, transA,
                        m, n1, n2,
                        alpha, dB(0, n1), lddb,
                               dA(0, n1), ldda,
                        c_one, dB(0,  0), lddb,
                        queue );

                magmablas_ztrmm_core(
                        side, uplo, transA, diag,
                        m, n2, alpha,
                        dA(n1, n1), ldda,
                        dB( 0, n1), lddb,
                        queue );
            }
            break;
        default:; // propose something
    }
#undef dA
#undef dB
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Purpose
    =======

    ZTRMM  performs one of the matrix-matrix operations

       B := alpha*op( A )*B,   or   B := alpha*B*op( A )

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).

    Parameters
    ==========

    @param[in]
    side     magma_side_t.
             On entry,  side specifies whether  op( A ) multiplies B from
             the left or right as follows:
                side = magmaLeft   B := alpha*op( A )*B.
                side = magmaRight  B := alpha*B*op( A ).
             Unchanged on exit.

    @param[in]
    uplo     magma_uplo_t.
             On entry, uplo specifies whether the matrix A is an upper or
             lower triangular matrix as follows:
                uplo = magmaUpper   A is an upper triangular matrix.
                uplo = magmaLower   A is a lower triangular matrix.
             Unchanged on exit.

    @param[in]
    transA   magma_trans_t.
             On entry, transA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:
                transA = MagmaNoTrans     op( A ) = A.
                transA = MagmaTrans       op( A ) = A'.
                transA = MagmaConjTrans   op( A ) = conjg( A' ).
             Unchanged on exit.

    @param[in]
    diag     magma_diag_t.
             On entry, diag specifies whether or not A is unit triangular
             as follows:
                diag = MagmaUnit      A is assumed to be unit triangular.
                diag = MagmaNonUnit   A is not assumed to be unit
                                    triangular.
             Unchanged on exit.

    @param[in]
    m        INTEGER.
             On entry, m specifies the number of rows of B. m must be at
             least zero.
             Unchanged on exit.

    @param[in]
    n        INTEGER.
             On entry, n specifies the number of columns of B.  n must be
             at least zero.
             Unchanged on exit.

    @param[in]
    alpha    DOUBLE COMPLEX.
             On entry,  alpha specifies the scalar  alpha. When  alpha is
             zero then  A is not referenced and  B need not be set before
             entry.
             Unchanged on exit.

    @param[in]
    dA       DOUBLE COMPLEX array A of DIMENSION ( ldda, k ), where k is m
             when  side = magmaLeft  and is  n  when  side = magmaRight.
             Before entry  with  uplo = magmaUpper,  the  leading  k by k
             upper triangular part of the array  A must contain the upper
             triangular matrix  and the strictly lower triangular part of
             A is not referenced.
             Before entry  with  uplo = magmaLower,  the  leading  k by k
             lower triangular part of the array  A must contain the lower
             triangular matrix  and the strictly upper triangular part of
             A is not referenced.
             Note that when  diag = MagmaUnit,  the diagonal elements of
             A  are not referenced either,  but are assumed to be  unity.
             Unchanged on exit.

    @param[in]
    ldda     INTEGER.
             On entry, ldda specifies the first dimension of A as declared
             in the calling (sub) program.  When  side = magmaLeft  then
             ldda  must be at least  max( 1, m ),  when  side = magmaRight
             then ldda must be at least max( 1, n ).
             Unchanged on exit.

    @param[in,out]
    dB_array DOUBLE COMPLEX array B of DIMENSION ( lddb, n ).
             Before entry,  the leading  m by n part of the array  B must
             contain the matrix  B,  and  on exit  is overwritten  by the
             transformed matrix.

    @param[in]
    lddb     INTEGER.
             On entry, lddb specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   lddb  must  be  at  least
             max( 1, m ).
             Unchanged on exit.

    @param[in]
    queue   magma_queue_t.
            Queue to execute in.

    @ingroup magma_trmm
    ===================================================================== */
extern "C" void
magmablas_ztrmm(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dB, magma_int_t lddb,
        magma_queue_t queue )
{
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
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( m <= 0 || n <= 0 )
        return;

    magmablas_ztrmm_core(
        side, uplo, transA, diag,
        m, n, alpha,
        dA, ldda, dB, lddb, queue );

}
///////////////////////////////////////////////////////////////////////////////////////////////////
