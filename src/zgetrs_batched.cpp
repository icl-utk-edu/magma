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
    with a general N-by-N matrix A using the LU factorization computed by ZGETRF.

    This is a batched version that solves batchCount N-by-N matrices in parallel.
    dA, dB, and ipiv become arrays with one entry per matrix.

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


    @param[out]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in,out]
    dB_array   Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDB,N).
            On entry, each pointer is an right hand side matrix B.
            On exit, each pointer is the solution matrix X.


    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).


    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.


    @ingroup magma_getrs_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrs_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t batchCount, magma_queue_t queue)
{
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

    magmaDoubleComplex* dwork        = NULL; // dwork is workspace for ztrsv
    magmaDoubleComplex **dwork_array = NULL;
    
    // batch trsv requires workspace
    if(nrhs == 1){ 
        magma_int_t dwork_msize = n*nrhs;    // TODO: resize workspace for trsv purpose only
        magma_malloc((void**)&dwork_array, batchCount * sizeof(*dwork_array));
        magma_zmalloc( &dwork, dwork_msize * batchCount );
        /* check allocation */
        if (dwork     == NULL  || dwork_array == NULL) {
            magma_free(dwork_array);
            magma_free( dwork );
            info = MAGMA_ERR_DEVICE_ALLOC;
            magma_xerbla( __func__, -(info) );
            return info;
        }
        magmablas_zlaset( MagmaFull, dwork_msize, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dwork, dwork_msize, queue );
        magma_zset_pointer( dwork_array, dwork, n, 0, 0, dwork_msize, batchCount, queue );
    }

    if (notran) {
        magma_zlaswp_rowserial_batched( nrhs, dB_array, lddb, 1, n, dipiv_array, batchCount, queue );

        if (nrhs > 1){
            // solve dwork = L^-1 * NRHS
            magmablas_ztrsm_batched( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
                    n, nrhs, MAGMA_Z_ONE, 
                    dA_array, ldda, 
                    dB_array, lddb, 
                    batchCount, queue);

            // solve X = U^-1 * dwork
            magmablas_ztrsm_batched( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    n, nrhs, MAGMA_Z_ONE, 
                    dA_array, ldda, 
                    dB_array, lddb, 
                    batchCount, queue);
        }
        else{
            // solve dwork = L^-1 * 1
            magmablas_ztrsv_outofplace_batched( MagmaLower, MagmaNoTrans, MagmaUnit,
                n, 
                dA_array,       ldda, // dA
                dB_array,        1, // dB
                dwork_array,     // dX //output
                batchCount, queue, 0 );

            // solve X = U^-1 * dwork
            magmablas_ztrsv_outofplace_batched( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                n, 
                dA_array,       ldda, // dA
                dwork_array,        1, // dB 
                dB_array,    // dX //output
                batchCount, queue, 0 );
        }
    }
    else {
        if (nrhs > 1){
            /* Solve A**T * X = B  or  A**H * X = B. */
            // solve 
            magmablas_ztrsm_batched( MagmaLeft, MagmaUpper, trans, MagmaUnit, 
                    n, nrhs, MAGMA_Z_ONE, 
                    dA_array, ldda, 
                    dB_array, lddb, 
                    batchCount, queue);
            
            // solve 
            magmablas_ztrsm_batched( MagmaLeft, MagmaLower, trans, MagmaNonUnit, 
                    n, nrhs, MAGMA_Z_ONE, 
                    dA_array, ldda, 
                    dB_array, lddb, 
                    batchCount, queue);
        }
        else{
            /* Solve A**T * X = B  or  A**H * X = B. */
            // solve 
            magmablas_ztrsv_outofplace_batched( MagmaUpper, trans, MagmaUnit, 
                n, 
                dA_array,       ldda, // dA
                dB_array,      1, // dB
                dwork_array,       // dX //output
                batchCount, queue, 0 );

            // solve 
            magmablas_ztrsv_outofplace_batched( MagmaLower, trans, MagmaNonUnit, 
                n, 
                dA_array,       ldda, // dA
                dwork_array,        1, // dB 
                dB_array,       // dX //output
                batchCount, queue, 0 );
        }

        magma_zlaswp_rowserial_batched( nrhs, dB_array, lddb, 1, n, dipiv_array, batchCount, queue );
    }

    magma_queue_sync(queue);

    if(nrhs == 1){
        magma_free(dwork_array);
        magma_free( dwork );
    }

    return info;
}
