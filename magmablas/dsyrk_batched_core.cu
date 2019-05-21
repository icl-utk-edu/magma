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

#define PRECISION_d

#include "herk_template_kernel_batched.cuh"
#include "gemm_config/dgemm_param_nn.h"
#include "gemm_config/dgemm_param_nt.h"
#include "gemm_config/dgemm_param_tn.h"
#include "gemm_config/dgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_dsyrk_batched_core(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    double const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    double beta,
    double **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    double cbeta  = MAGMA_D_MAKE( beta, 0. );
    double calpha = MAGMA_D_MAKE( alpha, 0. );

    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( n <= 0 || k <= 0 )
        return;

    // we have two shapes only (nt or tn)
    magma_int_t shape = 0;
    if      (trans == MagmaNoTrans)   { shape = 0; } // nt
    else                              { shape = 1; } // tn
    
    switch(shape)
    {
        case 0: // nt
            {
                if (k < 128)
                {
                    herk_template_batched_nt<double, version(NT,160), 0, 0>
                    (uplo, n, k, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, dC_array, ci, cj, lddc, calpha, cbeta, batchCount, queue);
                }
                else
                {
                    if (n < 256)
                    {
                        herk_template_batched_nt<double, version(NT,160), 0, 0>
                        (uplo, n, k, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, dC_array, ci, cj, lddc, calpha, cbeta, batchCount, queue);
                    }
                    else
                    {
                        herk_template_batched_nt<double, version(NT,190), 0, 0>
                        (uplo, n, k, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, dC_array, ci, cj, lddc, calpha, cbeta, batchCount, queue);
                    }
                }
            }
            break;
        case 1: // tn
            {
                if (k < 64)
                {
                    herk_template_batched_tn<double, version(TN,207), 0, 0>
                    (uplo, n, k, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, dC_array, ci, cj, lddc, calpha, cbeta, batchCount, queue);
                }
                else
                {
                    if (n < 256)
                    {
                        herk_template_batched_tn<double, version(TN,207), 0, 0>
                        (uplo, n, k, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, dC_array, ci, cj, lddc, calpha, cbeta, batchCount, queue);
                    }
                    else
                    {
                        herk_template_batched_tn<double, version(TN,209), 0, 0>
                        (uplo, n, k, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, dC_array, ci, cj, lddc, calpha, cbeta, batchCount, queue);
                    }
                }
            }
            break;
        default:; // propose something
    }
}


/***************************************************************************//**
    Purpose
    -------
    DSYRK performs one of the symmetric rank k operations

    C := alpha*A*A**H + beta*C,

    or

    C := alpha*A**H*A + beta*C,

    where alpha and beta are real scalars, C is an n by n symmetric
    matrix and A is an n by k matrix in the first case and a k by n
    matrix in the second case.
    
    Parameters
    ----------

    @param[in]
    uplo    magma_uplo_t.
           On entry, uplo specifies whether the upper or lower
           triangular part of the array C is to be referenced as
           follows:

           uplo = MagmaUpper Only the upper triangular part of C
           is to be referenced.

           uplo = MagmaLower Only the lower triangular part of C
           is to be referenced.
    
    @param[in]
    trans   magma_trans_t.
            On entry, trans specifies the operation to be performed as
            follows:

            trans = MagmaNoTrans C := alpha*A*A**H + beta*C.

            trans = MagmaTrans   C := alpha*A**H*A + beta*C.

    @param[in]
    n       INTEGER.
            On entry,  specifies the order of the matrix C. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry with trans = MagmaNoTrans, k specifies the number
            of columns of the matrix A, and on entry with
            trans = MagmaTrans, k specifies the number of rows of the
            matrix A. K must be at least zero.

    @param[in]
    alpha   DOUBLE PRECISION
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array A of DIMENSION ( ldda, ka ), where ka is
             k  when  trans = MagmaNoTrans,  and is  n  otherwise.
             Before entry with  trans = MagmaNoTrans,  the leading  m by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by m  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of each array A as declared
            in the calling (sub) program. When  trans = MagmaNoTrans then
            ldda must be at least  max( 1, n ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    beta    DOUBLE PRECISION.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.
    
    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array C of DIMENSION ( lddc, n ).
             Before entry with uplo = MagmaUpper, the leading n by n
             upper triangular part of the array C must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of C is not referenced. On exit, the
             upper triangular part of the array C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with uplo = MagmaLower, the leading n by n
             lower triangular part of the array C must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of C is not referenced. On exit, the
             lower triangular part of the array C is overwritten by the
             lower triangular part of the updated matrix.
             Note that the imaginary parts of the diagonal elements need
             not be set, they are assumed to be zero, and on exit they
             are set to zero.

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
    
    @ingroup magma_syrk_batched
*******************************************************************************/
extern "C" void
magmablas_dsyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double beta,
    double **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        printf("not supported \n"); // TODO call cublas
        return;
    }
    
    magmablas_dsyrk_batched_core(
            uplo, trans, 
            n, k, 
            alpha, dA_array, 0, 0, ldda, 
                   dA_array, 0, 0, ldda, 
            beta,  dC_array, 0, 0, lddc, batchCount, queue );
}
