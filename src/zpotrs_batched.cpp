
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
    ZPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by ZPOTRF.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA_array    Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
             The triangular factor U or L from the Cholesky factorization
             A = U**H*U or A = L*L**H, as computed by ZPOTRF.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,N).

    @param[in,out]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
             On entry, each pointer is a right hand side matrix B.
             On exit, the corresponding solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  LDDB >= max(1,N).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.


    @ingroup magma_potrs_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrs_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t batchCount, magma_queue_t queue)
{
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    if ( n < 0 )
        info = -2;
    if ( nrhs < 0)
        info = -3;
    if ( ldda < max(1, n) )
        info = -5;
    if ( lddb < max(1, n) )
        info = -7;
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return info;
    }
    
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dW4_displ  = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **dwork_array = NULL;

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
         dinvA_array == NULL || dwork_array == NULL || dinvA     == NULL || dwork     == NULL ) {
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

    if ( uplo == MagmaUpper) {
        if (nrhs > 1)
        {
            // A = U^T U
            // solve U^{T}X = B ==> dworkX = U^-T * B
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, 1,
                    n, nrhs,
                    c_one,
                    dA_array,       ldda, // dA
                    dB_array,      lddb, // dB
                    dwork_array,        n, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );

            // solve U X = dwork ==> X = U^-1 * dwork
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 1,
                    n, nrhs,
                    c_one,
                    dA_array,       ldda, // dA
                    dwork_array,        n, // dB 
                    dB_array,   lddb, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );
        }
        else
        {
            // A = U^T U
            // solve U^{T}X = B ==> dworkX = U^-T * B
            magmablas_ztrsv_outofplace_batched( MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                    n, 
                    dA_array,       ldda, // dA
                    dB_array,      1, // dB
                    dwork_array,     // dX //output
                    batchCount, queue, 0 );

            // solve U X = dwork ==> X = U^-1 * dwork
            magmablas_ztrsv_outofplace_batched( MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    n, 
                    dA_array,       ldda, // dA
                    dwork_array,        1, // dB 
                    dB_array,   // dX //output
                    batchCount, queue, 0 );
        }
    }
    else {
        if (nrhs > 1)
        {
            // A = L L^T
            // solve LX= B ==> dwork = L^{-1} B
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 1,
                    n, nrhs,
                    c_one,
                    dA_array,       ldda, // dA
                    dB_array,      lddb, // dB
                    dwork_array,        n, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );

            // solve L^{T}X= dwork ==> X = L^{-T} dwork
            magmablas_ztrsm_outofplace_batched( MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, 1,
                    n, nrhs,
                    c_one,
                    dA_array,       ldda, // dA
                    dwork_array,        n, // dB 
                    dB_array,   lddb, // dX //output
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );
        }
        else
        {
            // A = L L^T
            // solve LX= B ==> dwork = L^{-1} B
            magmablas_ztrsv_outofplace_batched( MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                    n,
                    dA_array,       ldda, // dA
                    dB_array,      1, // dB
                    dwork_array,   // dX //output
                    batchCount, queue, 0 );

            // solve L^{T}X= dwork ==> X = L^{-T} dwork
            magmablas_ztrsv_outofplace_batched( MagmaLower, MagmaConjTrans, MagmaNonUnit,
                    n,
                    dA_array,       ldda, // dA
                    dwork_array,        1, // dB 
                    dB_array,     // dX //output
                    batchCount, queue, 0 );
        }
    }

    magma_queue_sync(queue);

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
