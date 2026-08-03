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
#include "magma_internal.h"

#define PRECISION_z

#include "gemm_packed_template_kernel_batched.cuh"
#include "gemm_config/zgemm_param_nn.h"
#include "gemm_config/zgemm_param_tn.h"

#define version(s,v) s ## _V_ ## v

/***************************************************************************//**
    Purpose
    -------
    ZGEMM performs one of the matrix-matrix operations

        C = alpha*op( A )*B + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, B a k by n matrix and C an m by n matrix.
    The full A matrix is Hermitian or symmetric and stored in a packed
    format, where only the upper or lower portion is stored. This
    routine may ONLY be used when accessing a block of the matrix stored
    entirely in the lower (or upper) part, i.e., the portion that is
    explicitly stored. To apply the entire symmetric matrix from a packed
    format, see zhemm_packed.

    Parameters
    ----------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.

    @param[in]
    uplo    magma_uplo_t
            On entry, uplo specifies whether the upper or lower
            triangular part of the Hermitian matrix A stored:

            uplo = MagmaUpper   Only the upper triangular part of the
                                Hermitian matrix is stored.
            uplo = MagmaLower   Only the lower triangular part of the
                                Hermitian matrix is stored.

    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( A )  and of the  matrix C.  M  must  be at least  zero.

    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( B ) and the number of columns of the matrix C. N must be
            at least zero.

    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( A ) and the number of rows of the matrix op( B ). K must
            be at least  zero.

    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of DIMENSION ( ldda, ka ), where ka is
             k  when  transA = MagmaNoTrans,  and is  m  otherwise.
             Before entry with  transA = MagmaNoTrans,  the leading  m by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by m  part of the array A must contain  the
             matrix A.

    @param[in]
    Ai   INTEGER
            Row offset for all 'A' matrices.

    @param[in]
    Aj   INTEGER
            Column offset for all 'A' matrices.

    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of each array A as declared
            in the calling (sub) program. When  transA = MagmaNoTrans then
            ldda must be at least  max( 1, m ), otherwise  ldda must be at
            least  max( 1, k ).

    @param[in]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array B of DIMENSION ( lddb, kb ), where kb is
             n  when  transB = MagmaNoTrans,  and is  k  otherwise.
             Before entry with  transB = MagmaNoTrans,  the leading  k by n
             part of the array B must contain the matrix B, otherwise
             the leading  n by k  part of the array B must contain  the
             matrix B.

    @param[in]
    Bi   INTEGER
            Row offset for all 'B' matrices.

    @param[in]
    Bj   INTEGER
            Column offset for all 'B' matrices.

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B as declared
            in the calling (sub) program. When  transB = MagmaNoTrans then
            lddb must be at least  max( 1, k ), otherwise  lddb must be at
            least  max( 1, n ).

    @param[in]
    beta    COMPLEX_16.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array C of DIMENSION ( lddc, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

    @param[in]
    Ci   INTEGER
            Row offset for all 'C' matrices.

    @param[in]
    Cj   INTEGER
            Column offset for all 'C' matrices.

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of each array C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemm_batched
*******************************************************************************/
void
magmablas_zgemm_packed_batched_core(
    magma_trans_t transA, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( lddc < m )
        info = -13;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( m <= 0 || n <= 0 || k <= 0 )
        return;

    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && uplo == MagmaLower)   { shape = 0; } // nn with lower
    else if (transA == MagmaNoTrans   && uplo == MagmaUpper)   { shape = 1; } // nn with upper
    else if (transA == MagmaTrans     && uplo == MagmaLower)   { shape = 2; } // tn with lower
    else if (transA == MagmaTrans     && uplo == MagmaUpper)   { shape = 3; } // tn with upper
    else if (transA == MagmaConjTrans && uplo == MagmaLower)   { shape = 4; } // cn with lower
    else if (transA == MagmaConjTrans && uplo == MagmaUpper)   { shape = 5; } // cn with upper

    switch(shape)
    {
        case 0: // nn with lower
            {
                gemm_lower_packed_template_batched_nn<magmaDoubleComplex, version(NN,18), 0, 0>
                (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
            }
            break;
        case 1: // nn with upper
            {
                gemm_upper_packed_template_batched_nn<magmaDoubleComplex, version(NN,18), 0, 0>
                (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
            }
            break;
        case 2: // tn with lower
            {
                if(m == n && m < 32) {
                    if(m <= 8) {
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 8, 8, 32, 1, 8, 8, 8, 8, 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if(m <= 16) {
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 16, 4, 16, 16, 16, 1, 16, 4, 16, 4, 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if (m <= 24) {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24,  8, 1, 8, 8, 8, 8, 0, 0>
                        #else
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24, 16, 1, 8, 8, 8, 8, 0, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex,  8, 8, 32, 32,  8, 1,  8, 8,  8, 8, 0, 0>
                        #else
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 16, 8, 32, 32, 64, 1, 16, 8, 16, 8, 0, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                }
                else {
                    gemm_lower_packed_template_batched_tn<magmaDoubleComplex, version(TN,72), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                }
            }
            break;
        case 3: // tn with upper
            {
                if(m == n && m < 32) {
                    if(m <= 8) {
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 8, 8, 32, 1, 8, 8, 8, 8, 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if(m <= 16) {
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 16, 4, 16, 16, 16, 1, 16, 4, 16, 4, 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if (m <= 24) {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24,  8, 1, 8, 8, 8, 8, 0, 0>
                        #else
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24, 16, 1, 8, 8, 8, 8, 0, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex,  8, 8, 32, 32,  8, 1,  8, 8,  8, 8, 0, 0>
                        #else
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 16, 8, 32, 32, 64, 1, 16, 8, 16, 8, 0, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                }
                else {
                    gemm_upper_packed_template_batched_tn<magmaDoubleComplex, version(TN,72), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                }
            }
            break;
        case 4: // cn with lower
            {
                if(m == n && m < 32) {
                    if(m <= 8) {
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 8, 8, 32, 1, 8, 8, 8, 8, 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if(m <= 16) {
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 16, 4, 16, 16, 16, 1, 16, 4, 16, 4, 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if (m <= 24) {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24,  8, 1, 8, 8, 8, 8, 1, 0>
                        #else
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24, 16, 1, 8, 8, 8, 8, 1, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex,  8, 8, 32, 32,  8, 1,  8, 8,  8, 8, 1, 0>
                        #else
                        gemm_lower_packed_template_batched_tn<magmaDoubleComplex, 16, 8, 32, 32, 64, 1, 16, 8, 16, 8, 1, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                }
                else {
                    gemm_lower_packed_template_batched_tn<magmaDoubleComplex, version(TN,72), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                }
            }
            break;
        case 5: // cn with upper
            {
                if(m == n && m < 32) {
                    if(m <= 8) {
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 8, 8, 32, 1, 8, 8, 8, 8, 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if(m <= 16) {
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 16, 4, 16, 16, 16, 1, 16, 4, 16, 4, 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else if (m <= 24) {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24,  8, 1, 8, 8, 8, 8, 1, 0>
                        #else
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 8, 8, 24, 24, 16, 1, 8, 8, 8, 8, 1, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                    else {
                        #ifdef MAGMA_HAVE_HIP
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex,  8, 8, 32, 32,  8, 1,  8, 8,  8, 8, 1, 0>
                        #else
                        gemm_upper_packed_template_batched_tn<magmaDoubleComplex, 16, 8, 32, 32, 64, 1, 16, 8, 16, 8, 1, 0>
                        #endif
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                    }
                }
                else {
                    gemm_upper_packed_template_batched_tn<magmaDoubleComplex, version(TN,72), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
                }
            }
            break;
        default:; // propose something
    }
}
