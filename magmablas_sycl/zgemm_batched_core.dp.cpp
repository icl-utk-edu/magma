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
       @author Natalie Beams

*/
#include <sycl/sycl.hpp>
#include "magma_internal.h"

#define PRECISION_z

#include "magma_tuning_trees.h"
#include "gemm_template_kernel_batched.dp.hpp"
#include "gemm_config/zgemm_param_nn.h"
#include "gemm_config/zgemm_param_nt.h"
#include "gemm_config/zgemm_param_tn.h"
#include "gemm_config/zgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/***************************************************************************//**
    Purpose
    -------
    ZGEMM performs one of the matrix-matrix operations

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
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.

    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.

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
magmablas_zgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
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

    // special case for small square matrices
    if( m == n && n == k && m <= magma_get_zgemm_batched_smallsq_limit(m)){
        magmablas_zgemm_batched_smallsq(
                transA, transB,
                m, n, k,
                alpha, dA_array, Ai, Aj, ldda,
                       dB_array, Bi, Bj, lddb,
                beta,  dC_array, Ci, Cj, lddc, batchCount, queue );
        return;
    }

    magma_int_t config = magma_zgemm_batched_get_config(transA, transB, m, n, k);
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
   
    magma_int_t err = 0; 
    switch(shape)
    {
        case 0: // nn
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,0), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 10:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,10), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 11:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,11), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 19:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,19), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 21:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,21), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 22:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,22), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,3), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,6), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 7:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,7), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 9:
		        {
		          gemm_template_batched_nn<magmaDoubleComplex, version(NN,9), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 1: // nt
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,0), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 14:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,14), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 16:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,16), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 18:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,18), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,3), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 33:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,33), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,6), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 64:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,64), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 68:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,68), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 8:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,8), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 2: // nc
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,0), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 14:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,14), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 16:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,16), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 18:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,18), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,3), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 33:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,33), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,6), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 64:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,64), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 68:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,68), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 8:
		        {
		          gemm_template_batched_nt<magmaDoubleComplex, version(NT,8), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 3: // tn
            {
		switch(config)
		{
		    case 13:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,13), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 148:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,148), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 22:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,22), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 28:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,28), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,3), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 44:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,44), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 45:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,45), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 47:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,47), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 66:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,66), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 69:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,69), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 6: // cn
            {
		switch(config)
		{
		    case 13:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,13), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 148:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,148), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 22:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,22), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 28:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,28), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,3), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 44:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,44), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 45:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,45), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 47:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,47), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 66:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,66), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 69:
		        {
		          gemm_template_batched_tn<magmaDoubleComplex, version(TN,69), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 4: // tt
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,0), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 11:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,11), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 13:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,13), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 14:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,14), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 17:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,17), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 19:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,19), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,3), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 5:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,5), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 53:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,53), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,6), 0, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 5: // tc
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,0), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 11:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,11), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 13:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,13), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 14:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,14), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 17:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,17), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 19:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,19), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,3), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 5:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,5), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 53:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,53), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,6), 0, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 7: // ct
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,0), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 11:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,11), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 13:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,13), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 14:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,14), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 17:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,17), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 19:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,19), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,3), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 5:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,5), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 53:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,53), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,6), 1, 0>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        case 8: // cc
            {
		switch(config)
		{
		    case 0:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,0), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 11:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,11), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 13:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,13), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 14:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,14), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 17:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,17), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 19:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,19), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 3:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,3), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 5:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,5), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 53:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,53), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
		    case 6:
		        {
		          gemm_template_batched_tt<magmaDoubleComplex, version(TT,6), 1, 1>
                          (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue);
			}
			break;
                    default:
		        err = 1;
		}
		if (err > 0)
	            printf("Error determining batched GEMM configuration!\n");
            }
            break;
        default:; // propose something
    }
}
