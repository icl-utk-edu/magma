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
#include "trmv_template_kernel.cuh"

magma_int_t magma_get_ztrmv_nb(magma_int_t n)
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
magmablas_ztrmv_small(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dX, magma_int_t incx,
        magma_queue_t queue )
{
    if(transA == MagmaConjTrans) {
        trmv_template<magmaDoubleComplex, ZTRMV_NB, 1>
        (uplo, transA, diag, n, dA, ldda, dX, incx, queue);
    }
    else {
        trmv_template<magmaDoubleComplex, ZTRMV_NB, 0>
        (uplo, transA, diag, n, dA, ldda, dX, incx, queue);
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void
magmablas_ztrmv_core(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dX, magma_int_t incx,
        magma_queue_t queue )
{
#define dA(i,j) dA + j*ldda + i
#define dX(j)   dX + j*incx

    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    // stopping condition
    if(nrowA <= ZTRMV_NB){
        magmablas_ztrmv_small( uplo, transA, diag, n, dA, ldda, dX, incx, queue );
        return;
    }

    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // NL
    else if (transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // NU
    else if (transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // TL | CL
    else if (transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // TU | CU

    // at this point we can tell that nrowA > ZTRMV_NB
    switch(shape)
    {
        case 0: // Nl
            {
                const int m1 = magma_get_ztrmv_nb(m);
                const int m2 = m - m1;

                magmablas_ztrmv_core(
                        uplo, transA, diag, m2,
                        dA(m1, m1), ldda, dX(m1), incx, queue );

                magmablas_zgemv(
                        MagmaNoTrans, m2, m1,
                        c_one, dA(m1, 0), ldda,
                               dX( 0), incx,
                        c_one, dX(m1), incx,
                        queue );

                magmablas_ztrmv_core(
                        uplo, transA, diag, m1,
                        dA(0, 0), ldda, dX(0), incx, queue );
            }
            break;
        case 1: // NU
            {
                const int m2 = magma_get_ztrmv_nb(m);
                const int m1 = m - m2;

                magmablas_ztrmv_core(
                        uplo, transA, diag, m1,
                        dA(0, 0), ldda, dX(0), incx, queue );

                magma_zgemv(
                        MagmaNoTrans,
                        m1, m2,
                        c_one, dA( 0, m1), ldda,
                               dX(m1), incx,
                        c_one, dX( 0), incx,
                        queue );

                magmablas_ztrmv_core(
                        uplo, transA, diag, m2,
                        dA(m1, m1), ldda, dX(m1), incx, queue );
            }
            break;
        case 2: // TL || CL
            {
                const int m2 = magma_get_ztrmv_nb(m);
                const int m1 = m - m2;

                magmablas_ztrmv_core(
                        uplo, transA, diag, m1,
                        dA(0, 0), ldda, dX(0), incx, queue );

                magma_zgemv(
                        transA,
                        m2, m1,
                        c_one, dA(m1, 0), ldda,
                               dX(m1), incx,
                        c_one, dX( 0), incx,
                        queue );

                magmablas_ztrmv_core(
                        uplo, transA, diag, m2,
                        dA(m1, m1), ldda, dX(m1), incx, queue );
            }
            break;
        case 3: // TU | CU
            {
                const int m1 = magma_get_ztrmv_nb(m);
                const int m2 = m - m1;

                magmablas_ztrmv_core(
                        uplo, transA, diag, m2,
                        dA(m1, m1), ldda, dX(m1), incx, queue );

                magma_zgemm(
                        transA,
                        m1, m2,
                        c_one, dA(0, m1), ldda,
                               dX( 0), incx,
                        c_one, dX(m1), incx,
                        queue );

                magmablas_ztrmv_core(
                        uplo, transA, diag, m1,
                        dA(0, 0), ldda, dX(0), incx, queue );
            }
            break;
        default:; // propose something
    }
#undef dA
#undef dX
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
    dA_array     Array of pointers, dimension(batchCount).
             Each is a DOUBLE COMPLEX array A of DIMENSION ( ldda, k ), where k is m
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
    dB_array     Array of pointers, dimension(batchCount).
             Each is a DOUBLE COMPLEX array B of DIMENSION ( lddb, n ).
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
    batchCount  INTEGER.
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t.
            Queue to execute in.

    @ingroup magma_trmm_batched
    ===================================================================== */
extern "C" void
magmablas_ztrmv(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dX, magma_int_t incx,
        magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -4;
    } else if (ldda < max(1,n)) {
        info = -6;
    } else if (incx < 0) {    // no support yet for a negative incx
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( n <= 0 )
        return;

    magmablas_ztrmv_core( uplo, transA, diag, n, dA, ldda, dX, incx, queue );

}
///////////////////////////////////////////////////////////////////////////////////////////////////
