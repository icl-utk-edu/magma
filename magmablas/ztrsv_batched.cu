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
#include "batched_kernel_param.h"

#define PRECISION_z

#include "gemm_template_device_defs.cuh"
#include "trsv_template_device.cuh"
#include "trsv_template_kernel_batched.cuh"

/******************************************************************************/
static void
magmablas_ztrsv_small_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dx_array, magma_int_t xi, magma_int_t incx,
        magma_int_t batchCount, magma_queue_t queue )
{
    if     ( n <=  2 )
        trsv_small_batched<magmaDoubleComplex,  2>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <=  4 )
        trsv_small_batched<magmaDoubleComplex,  4>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <=  8 )
        trsv_small_batched<magmaDoubleComplex,  8>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <= 16 )
        trsv_small_batched<magmaDoubleComplex, 16>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <= 32 )
        trsv_small_batched<magmaDoubleComplex, 32>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else
        printf("error in function %s: nrowA must be less than 32\n", __func__);
}

/******************************************************************************/
static magma_int_t magma_get_ztrsv_batched_nb(magma_int_t n)
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
magmablas_ztrsv_recursive_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dx_array, magma_int_t xi, magma_int_t incx,
        magma_int_t batchCount, magma_queue_t queue )
{
#define dA_array(i,j) dA_array, i, j
#define dx_array(i)   dx_array, i

    const magmaDoubleComplex c_one    = MAGMA_Z_ONE;
    const magmaDoubleComplex c_negone = MAGMA_Z_NEG_ONE;

    magma_int_t shape = -1;
    if      (transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // NL
    else if (transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // NU
    else if (transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // TL | CL
    else if (transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // TU | CU

    // stopping condition
    if(n <= 32){
        magmablas_ztrsv_small_batched(
            uplo, transA, diag, n,
            dA_array(Ai, Aj), ldda,
            dx_array(xi), incx, batchCount, queue );
        return;
    }

    const int n2 = magma_get_ztrsv_batched_nb(n);
    const int n1 = n - n2;

    switch(shape) {
        case 0: // Nl
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n2, n1,
                c_negone, dA_array(Ai+n1, Aj), ldda,
                          dx_array(xi       ), incx,
                c_one,    dx_array(xi+n1    ), incx,
                batchCount, queue );

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 1: // NU
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n1, n2,
                c_negone, dA_array(Ai, Aj+n1), ldda,
                          dx_array(xi+n1    ), incx,
                c_one,    dx_array(xi       ), incx,
                batchCount, queue );

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 2: // TL || CL
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n2, n1,
                c_negone, dA_array(Ai+n1, Aj), ldda,
                          dx_array(xi+n1    ), incx,
                c_one,    dx_array(xi       ), incx,
                batchCount, queue );


            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 3: // TU | lCU
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n1, n2,
                c_negone, dA_array(Ai, Aj+n1), ldda,
                          dx_array(xi       ), incx,
                c_one,    dx_array(xi+n1    ), incx,
                batchCount, queue );

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        default:; // propose something
    }
#undef dA_array
#undef dx_array
}

/***************************************************************************//**
    Purpose
    -------
    ztrsv solves one of the matrix equations on gpu

        op(A)*x = b,   or
        x*op(A) = b,

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
    A_array       Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array A of dimension ( lda, n ),
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
    lda     INTEGER.
            On entry, lda specifies the first dimension of A.
            lda >= max( 1, n ).

    @param[in]
    b_array     Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension  n
            On exit, b is overwritten with the solution vector X.

    @param[in]
    incb    INTEGER.
            On entry,  incb specifies the increment for the elements of
            b. incb must not be zero.
            Unchanged on exit.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trsv_batched
*******************************************************************************/
extern "C" void
magmablas_ztrsv_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex **A_array, magma_int_t lda,
    magmaDoubleComplex **b_array, magma_int_t incb,
    magma_int_t batchCount,
    magma_queue_t queue)
{
    magmablas_ztrsv_recursive_batched(
        uplo, trans, diag, n,
        A_array, 0, 0, lda,
        b_array, 0, incb,
        batchCount, queue );
}
