/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_z

#include "gemm_template_device_defs.cuh"
#include "trsv_template_device.cuh"
#include "trsv_template_kernel.cuh"


/******************************************************************************/
static void
magmablas_ztrsv_small(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dx, magma_int_t incx, magma_queue_t queue )
{
    if     ( n <=  2 )
        trsv_small<magmaDoubleComplex,  2>(uplo, transA, diag, n, dA, ldda, dx, incx, queue );
    else if( n <=  4 )
        trsv_small<magmaDoubleComplex,  4>(uplo, transA, diag, n, dA, ldda, dx, incx, queue );
    else if( n <=  8 )
        trsv_small<magmaDoubleComplex,  8>(uplo, transA, diag, n, dA, ldda, dx, incx, queue );
    else if( n <= 16 )
        trsv_small<magmaDoubleComplex, 16>(uplo, transA, diag, n, dA, ldda, dx, incx, queue );
    else if( n <= 32 )
        trsv_small<magmaDoubleComplex, 32>(uplo, transA, diag, n, dA, ldda, dx, incx, queue );
    else
        printf("error in function %s: nrowA must be less than 32\n", __func__);
}

/******************************************************************************/
static magma_int_t magma_get_ztrsv_nb(magma_int_t n)
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
static void
magmablas_ztrsv_recursive(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t ldda,
        magmaDoubleComplex *dx, magma_int_t incx, magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dx(i_)     (dx + (i_) * incx )

    const magmaDoubleComplex c_one    = MAGMA_Z_ONE;
    const magmaDoubleComplex c_negone = MAGMA_Z_NEG_ONE;

    magma_int_t shape = -1;
    if      (transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // NL
    else if (transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // NU
    else if (transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // TL | CL
    else if (transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // TU | CU

    // stopping condition
    if(n <= 32){
        magmablas_ztrsv_small(uplo, transA, diag, n, dA, ldda, dx, incx, queue );
        return;
    }

    const int n2 = magma_get_ztrsv_nb(n);
    const int n1 = n - n2;

    switch(shape) {
        case 0: // Nl
        {
            magmablas_ztrsv_recursive(
                uplo, transA, diag, n1,
                dA(0, 0), ldda,
                dx(0    ), incx, queue );

            magma_zgemv(
                transA, n2, n1,
                c_negone, dA(n1, 0), ldda,
                          dx(0    ), incx,
                c_one,    dx(n1   ), incx, queue );

            magmablas_ztrsv_recursive(
                uplo, transA, diag, n2,
                dA(n1, n1), ldda,
                dx(n1    ), incx, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 1: // NU
        {
            magmablas_ztrsv_recursive(
                uplo, transA, diag, n2,
                dA(n1, n1), ldda,
                dx(n1    ), incx, queue );

            magma_zgemv(
                transA, n1, n2,
                c_negone, dA(0, n1), ldda,
                          dx(n1   ), incx,
                c_one,    dx(0    ), incx, queue );

            magmablas_ztrsv_recursive(
                uplo, transA, diag, n1,
                dA(0, 0), ldda,
                dx(0   ), incx, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 2: // TL || CL
        {
            magmablas_ztrsv_recursive(
                uplo, transA, diag, n2,
                dA(n1, n1), ldda,
                dx(n1    ), incx, queue );

            magma_zgemv(
                transA, n2, n1,
                c_negone, dA(n1, 0), ldda,
                          dx(n1   ), incx,
                c_one,    dx(0    ), incx, queue );


            magmablas_ztrsv_recursive(
                uplo, transA, diag, n1,
                dA(0, 0), ldda,
                dx(0   ), incx, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 3: // TU | lCU
        {
            magmablas_ztrsv_recursive(
                uplo, transA, diag, n1,
                dA(0, 0), ldda,
                dx(0   ), incx, queue );

            magma_zgemv(
                transA, n1, n2,
                c_negone, dA(0, n1), ldda,
                          dx(0    ), incx,
                c_one,    dx(n1   ), incx, queue );

            magmablas_ztrsv_recursive(
                uplo, transA, diag, n2,
                dA(n1, n1), ldda,
                dx(n1    ), incx, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        default:; // propose something
    }
#undef dA
#undef dx
}

/***************************************************************************//**
    Purpose
    -------
    ztrsv solves one of the matrix equations on gpu

        op(A)*x = B,   or
        x*op(A) = B,

    where alpha is a scalar, X and B are vectors, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The vector x is overwritten on b.

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    trans  magma_trans_t.
            On entry, trans specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n N specifies the order of the matrix A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array of dimension ( lda, n )
            Before entry with uplo = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    ldda    INTEGER.
            On entry, lda specifies the first dimension of A.
            lda >= max( 1, n ).

    @param[in]
    db      COMPLEX_16 array of dimension  n
            On exit, b is overwritten with the solution vector X.

    @param[in]
    incb    INTEGER.
            On entry,  incb specifies the increment for the elements of
            b. incb must not be zero.
            Unchanged on exit.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trsv
*******************************************************************************/
extern "C" void
magmablas_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr db, magma_int_t incb,
    magma_queue_t queue)
{
    magmablas_ztrsv_recursive( uplo, trans, diag, n, (magmaDoubleComplex*)dA, ldda, db, incb, queue );
}
