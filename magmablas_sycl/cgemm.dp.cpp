/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define PRECISION_c

#include "gemm_template_kernel.dp.hpp"

#define version(s,v) s ## _V_ ## v

// TODO: further tuning
#if defined(MAGMA_HAVE_CUDA)
    #define DIM_X  16
    #define DIM_Y  16

    // nn
    #define BLK_M_nn   64
    #define BLK_N_nn   64
    #define BLK_K_nn   16
    #define DIM_XA_nn  32
    #define DIM_YA_nn  8
    #define DIM_XB_nn  16
    #define DIM_YB_nn  16

    // nt, nc
    #define BLK_M_nt   64
    #define BLK_N_nt   64
    #define BLK_K_nt   16
    #define DIM_XA_nt  16
    #define DIM_YA_nt  16
    #define DIM_XB_nt  16
    #define DIM_YB_nt  16

    // tn, cn
    #define BLK_M_tn   64
    #define BLK_N_tn   64
    #define BLK_K_tn   16
    #define DIM_XA_tn  16
    #define DIM_YA_tn  16
    #define DIM_XB_tn  16
    #define DIM_YB_tn  16

    // tt, tc, ct, cc
    #define BLK_M_tt   64
    #define BLK_N_tt   64
    #define BLK_K_tt   16
    #define DIM_XA_tt  16
    #define DIM_YA_tt  16
    #define DIM_XB_tt  32
    #define DIM_YB_tt  8

#else
    #define DIM_X  16
    #define DIM_Y  16

    // nn
    #define BLK_M_nn   64
    #define BLK_N_nn   32
    #define BLK_K_nn   8
    #define DIM_XA_nn  32
    #define DIM_YA_nn  8
    #define DIM_XB_nn  8
    #define DIM_YB_nn  32

    // nt, nc
    #define BLK_M_nt   64
    #define BLK_N_nt   64
    #define BLK_K_nt   16
    #define DIM_XA_nt  16
    #define DIM_YA_nt  16
    #define DIM_XB_nt  16
    #define DIM_YB_nt  16

    // tn, cn
    #define BLK_M_tn   64
    #define BLK_N_tn   64
    #define BLK_K_tn   16
    #define DIM_XA_tn  16
    #define DIM_YA_tn  16
    #define DIM_XB_tn  16
    #define DIM_YB_tn  16

    // tt, tc, ct, cc
    #define BLK_M_tt   64
    #define BLK_N_tt   64
    #define BLK_K_tt   16
    #define DIM_XA_tt  16
    #define DIM_YA_tt  16
    #define DIM_XB_tt  32
    #define DIM_YB_tt  8

#endif

/***************************************************************************//**
    Purpose
    -------
    CGEMM performs one of the matrix-matrix operations

        C = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.

    Parameters
    ----------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -      = MagmaNoTrans:   op( A ) = A.
      -      = MagmaTrans:     op( A ) = A**T.
      -      = MagmaConjTrans: op( A ) = A**H.

    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -      = MagmaNoTrans:   op( B ) = B.
      -      = MagmaTrans:     op( B ) = B**T.
      -      = MagmaConjTrans: op( B ) = B**H.

    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( dA )  and of the  matrix dC.  M  must  be at least  zero.

    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( dB ) and the number of columns of the matrix dC. N must be
            at least zero.

    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( dA ) and the number of rows of the matrix op( dB ). K must
            be at least  zero.

    @param[in]
    alpha   COMPLEX
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      COMPLEX array of DIMENSION ( LDA, ka ), where ka is
            k  when  transA = MagmaNoTrans,  and is  m  otherwise.
            Before entry with  transA = MagmaNoTrans,  the leading  m by k
            part of the array dA must contain the matrix dA, otherwise
            the leading  k by m  part of the array dA must contain  the
            matrix dA.

    @param[in]
    ldda    INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. When  transA = MagmaNoTrans then
            LDA must be at least  max( 1, m ), otherwise  LDA must be at
            least  max( 1, k ).

    @param[in]
    dB      COMPLEX array of DIMENSION ( LDB, kb ), where kb is
            n  when  transB = MagmaNoTrans,  and is  k  otherwise.
            Before entry with  transB = MagmaNoTrans,  the leading  k by n
            part of the array dB must contain the matrix dB, otherwise
            the leading  n by k  part of the array dB must contain  the
            matrix dB.

    @param[in]
    lddb    INTEGER.
            On entry, LDB specifies the first dimension of dB as declared
            in the calling (sub) program. When  transB = MagmaNoTrans then
            LDB must be at least  max( 1, k ), otherwise  LDB must be at
            least  max( 1, n ).

    @param[in]
    beta    COMPLEX.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then dC need not be set on input.

    @param[in,out]
    dC      COMPLEX array of DIMENSION ( LDC, n ).
            Before entry, the leading  m by n  part of the array  dC must
            contain the matrix  dC,  except when  beta  is zero, in which
            case dC need not be set on entry.
            On exit, the array  dC  is overwritten by the  m by n  matrix
            ( alpha*op( dA )*op( dB ) + beta*dC ).

    @param[in]
    lddc    INTEGER.
            On entry, LDC specifies the first dimension of dC as declared
            in  the  calling  (sub)  program.   LDC  must  be  at  least
            max( 1, m ).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemm
*******************************************************************************/
void
magmablas_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * dA, magma_int_t ldda,
    magmaFloatComplex const * dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex *dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( m <= 0 || n <= 0 || k <= 0 )
        return;

    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc

    switch(shape)
    {
        case 0: // nn
        {
            gemm_template_nn
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_nn, BLK_N_nn, BLK_K_nn, DIM_XA_nn, DIM_YA_nn, DIM_XB_nn, DIM_YB_nn, 0, 0>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 1: // nt
        {
            gemm_template_nt
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_nt, BLK_N_nt, BLK_K_nt, DIM_XA_nt, DIM_YA_nt, DIM_XB_nt, DIM_YB_nt, 0, 0>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 2: // nc
        {
            gemm_template_nt
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_nt, BLK_N_nt, BLK_K_nt, DIM_XA_nt, DIM_YA_nt, DIM_XB_nt, DIM_YB_nt, 0, 1>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 3: // tn
        {
            gemm_template_tn
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_tn, BLK_N_tn, BLK_K_tn, DIM_XA_tn, DIM_YA_tn, DIM_XB_tn, DIM_YB_tn, 0, 0>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 6: // cn
        {
            gemm_template_tn
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_tn, BLK_N_tn, BLK_K_tn, DIM_XA_tn, DIM_YA_tn, DIM_XB_tn, DIM_YB_tn, 1, 0>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 4: // tt
        {
            gemm_template_tt
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_tt, BLK_N_tt, BLK_K_tt, DIM_XA_tt, DIM_YA_tt, DIM_XB_tt, DIM_YB_tt, 0, 0>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 5: // tc
        {
            gemm_template_tt
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_tt, BLK_N_tt, BLK_K_tt, DIM_XA_tt, DIM_YA_tt, DIM_XB_tt, DIM_YB_tt, 0, 1>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 7: // ct
        {
            gemm_template_tt
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_tt, BLK_N_tt, BLK_K_tt, DIM_XA_tt, DIM_YA_tt, DIM_XB_tt, DIM_YB_tt, 1, 0>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        case 8: // cc
        {
            gemm_template_tt
            <magmaFloatComplex, DIM_X, DIM_Y, BLK_M_tt, BLK_N_tt, BLK_K_tt, DIM_XA_tt, DIM_YA_tt, DIM_XB_tt, DIM_YB_tt, 1, 1>
            (m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
        }
        break;
        default:; // propose something
    }
}
