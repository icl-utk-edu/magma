/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETRS solves a system of linear equations
        A * X = B,  A**T * X = B,  or  A**H * X = B
    with a general N-by-N matrix A using the LU factorization without pivoting computed by ZGETRF_NOPIV.

    This is a batched version that solves batchCount N-by-N matrices in parallel.
    dA, dB, become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A    * X = B  (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).


    @param[in,out]
    dB_array   Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDB,N).
            On entry, each pointer is an right hand side matrix B.
            On exit, each pointer is the solution matrix X.


    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.


    @ingroup magma_getrs_nopiv_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrs_nopiv_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount,  magma_queue_t queue)
{
    /* Local variables */

    magma_int_t notran = (trans == MagmaNoTrans);

    magma_int_t info = 0;
    if ( (! notran) &&
         (trans != MagmaTrans) &&
         (trans != MagmaConjTrans) ) {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (nrhs < 0) {
        info = -3;
    } else if (ldda < max(1,n)) {
        info = -5;
    } else if (lddb < max(1,n)) {
        info = -8;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }


    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return info;
    }

    magmaDoubleComplex **dA_displ   = NULL;
    magmaDoubleComplex **dB_displ  = NULL;
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dW4_displ  = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **dwork_array = NULL;

    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dB_displ,  batchCount * sizeof(*dB_displ));
    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dW4_displ,  batchCount * sizeof(*dW4_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dwork_array, batchCount * sizeof(*dwork_array));

    magma_int_t invA_msize = magma_roundup( n, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    magma_int_t dwork_msize = n*nrhs;
    magmaDoubleComplex* dinvA      = NULL;
    magmaDoubleComplex* dwork      = NULL; // dinvA and dwork are workspace in ztrsm
    magma_zmalloc( &dinvA, invA_msize * batchCount);
    magma_zmalloc( &dwork, dwork_msize * batchCount );
    /* check allocation */
    if ( dW1_displ == NULL || dW2_displ == NULL || dW3_displ   == NULL || dW4_displ   == NULL || 
         dinvA_array == NULL || dwork_array == NULL || dinvA     == NULL || dwork     == NULL ||
         dA_displ == NULL || dB_displ == NULL ) {
        magma_free(dA_displ);
        magma_free(dB_displ);
        magma_free(dW1_displ);
        magma_free(dW2_displ);
        magma_free(dW3_displ);
        magma_free(dW4_displ);
        magma_free(dinvA_array);
        magma_free(dwork_array);
        magma_free( dinvA );
        magma_free( dwork );
        info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magmablas_zlaset( MagmaFull, invA_msize, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA, invA_msize, queue );
    magmablas_zlaset( MagmaFull, dwork_msize, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dwork, dwork_msize, queue );
    magma_zset_pointer( dwork_array, dwork, n, 0, 0, dwork_msize, batchCount, queue );
    magma_zset_pointer( dinvA_array, dinvA, ZTRTRI_BATCHED_NB, 0, 0, invA_msize, batchCount, queue );

    magma_zdisplace_pointers(dA_displ, dA_array, ldda, 0, 0, batchCount, queue);
    magma_zdisplace_pointers(dB_displ, dB_array, lddb, 0, 0, batchCount, queue);

    if (notran) {
        if (nrhs > 1)
        {
            // solve dwork = L^-1 * NRHS
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 1,
                    n, nrhs,
                    MAGMA_Z_ONE,
                    dA_displ,       ldda, // dA
                    dB_displ,      lddb, // dB
                    dwork_array,        n, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );

            // solve X = U^-1 * dwork
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 1,
                    n, nrhs,
                    MAGMA_Z_ONE,
                    dA_displ,       ldda, // dA
                    dwork_array,        n, // dB 
                    dB_displ,   lddb, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );
        }
        else
        {
            // solve dwork = L^-1 * NRHS
            magmablas_ztrsv_outofplace_batched( MagmaLower, MagmaNoTrans, MagmaUnit, 
                    n, 
                    dA_displ,       ldda, // dA
                    dB_displ,      1, // dB
                    dwork_array,   // dX //output
                    batchCount, queue, 0 );

            // solve X = U^-1 * dwork
            magmablas_ztrsv_outofplace_batched( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    n, 
                    dA_displ,       ldda, // dA
                    dwork_array,        1, // dB 
                    dB_displ,   // dX //output
                    batchCount, queue, 0 );
        }
    }
    else {
        if (nrhs > 1)
        {
            /* Solve A**T * X = B  or  A**H * X = B. */
            // solve 
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaUpper, trans, MagmaUnit, 1,
                    n, nrhs,
                    MAGMA_Z_ONE,
                    dA_displ,       ldda, // dA
                    dB_displ,      lddb, // dB
                    dwork_array,        n, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );

            // solve 
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaLower, trans, MagmaNonUnit, 1,
                    n, nrhs,
                    MAGMA_Z_ONE,
                    dA_displ,       ldda, // dA
                    dwork_array,        n, // dB 
                    dB_displ,   lddb, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );
        }
        else
        {
            /* Solve A**T * X = B  or  A**H * X = B. */
            // solve 
            magmablas_ztrsv_outofplace_batched( MagmaUpper, trans, MagmaUnit, 
                    n, 
                    dA_displ,       ldda, // dA
                    dB_displ,      1, // dB
                    dwork_array,   // dX //output
                    batchCount, queue, 0 );
            // solve 
            magmablas_ztrsv_outofplace_batched( MagmaLower, trans, MagmaNonUnit,
                    n, 
                    dA_displ,       ldda, // dA
                    dwork_array,       1, // dB 
                    dB_displ,   // dX //output
                    batchCount, queue, 0 );
        }
    }

    magma_queue_sync(queue);

    magma_free(dA_displ);
    magma_free(dB_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dW4_displ);
    magma_free(dinvA_array);
    magma_free(dwork_array);
    magma_free( dinvA );
    magma_free( dwork );

    return info;
}
