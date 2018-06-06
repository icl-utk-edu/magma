
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
    
    magmaDoubleComplex **dwork_array = NULL;
    magmaDoubleComplex* dwork        = NULL; //dwork is workspace for ztrsv 
    
    if(nrhs == 1){
        magma_int_t dwork_msize = n*nrhs;        // TODO: resize dwork for trsv purpose only
        magma_malloc((void**)&dwork_array, batchCount * sizeof(*dwork_array));
        magma_zmalloc( &dwork, dwork_msize * batchCount );
        /* check allocation */
        if ( dwork_array == NULL || dwork     == NULL ) {
            magma_free(dwork_array);
            magma_free( dwork );
            info = MAGMA_ERR_DEVICE_ALLOC;
            magma_xerbla( __func__, -(info) );
            return info;
        }
        magmablas_zlaset( MagmaFull, dwork_msize, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dwork, dwork_msize, queue );
        magma_zset_pointer( dwork_array, dwork, n, 0, 0, dwork_msize, batchCount, queue );
    }

    if ( uplo == MagmaUpper) {
        if (nrhs > 1){
            // A = U^T U
            // solve U^{T} Y = B, where Y = U X 
            magmablas_ztrsm_batched(
                    MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, 
                    n, nrhs, c_one, 
                    dA_array, ldda, 
                    dB_array, lddb, batchCount, queue );

            // solve U X = B
            magmablas_ztrsm_batched(
                    MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    n, nrhs, c_one, 
                    dA_array, ldda, 
                    dB_array, lddb, batchCount, queue );
        }
        else{
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
        if (nrhs > 1){
            // A = L L^T
            // solve LY=B, where Y = L^{T} X
            magmablas_ztrsm_batched(
                    MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                    n, nrhs, 
                    c_one, 
                    dA_array, ldda, 
                    dB_array, lddb, batchCount, queue );

            // solve L^{T}X=B
            magmablas_ztrsm_batched(
                    MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    n, nrhs, c_one, 
                    dA_array, ldda, 
                    dB_array, lddb, batchCount, queue );
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

    if(nrhs == 1){
        magma_free(dwork_array);
        magma_free( dwork );
    }

    return info;
}
