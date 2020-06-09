/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ichi Yamazaki

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "trace.h"

/***************************************************************************//**
    Purpose
    =======
 
    ZHETRF computes the factorization of a complex Hermitian matrix A
    using the Bunch-Kaufman diagonal pivoting method.  The form of the
    factorization is
 
        A = U*D*U^H  or  A = L*D*L^H
 
    where U (or L) is a product of permutation and unit upper (lower)
    triangular matrices, and D is Hermitian and block diagonal with
    1-by-1 and 2-by-2 diagonal blocks.

    This is the blocked version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
 
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.
  
    @param[in,out]
    A       COMPLEX*16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, the block diagonal matrix D and the multipliers used
            to obtain the factor U or L (see below for further details).
 
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
 
    @param[out]
    ipiv    INTEGER array, dimension (N)
            Details of the interchanges and the block structure of D.
            If IPIV(k) > 0, then rows and columns k and IPIV(k) were
            interchanged and D(k,k) is a 1-by-1 diagonal block.
            If UPLO = MagmaUpper and IPIV(k) = IPIV(k-1) < 0, then rows and
            columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
            is a 2-by-2 diagonal block.  If UPLO = MagmaLower and IPIV(k) =
            IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
            interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, D(i,i) is exactly zero.  The factorization
                  has been completed, but the block diagonal matrix D is
                  exactly singular, and division by zero will occur if it
                  is used to solve a system of equations.

    Further Details
    ===============
    If UPLO = MagmaUpper, then A = U*D*U', where
    U = P(n)*U(n)* ... *P(k)U(k)* ...,
    i.e., U is a product of terms P(k)*U(k), where k decreases from n to
    1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
    and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
    defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
    that if the diagonal block D(k) is of order s (s = 1 or 2), then
 
               (   I    v    0   )   k-s
       U(k) =  (   0    I    0   )   s
               (   0    0    I   )   n-k
                  k-s   s   n-k
 
    If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
    If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
    and A(k,k), and v overwrites A(1:k-2,k-1:k).
  
    If UPLO = MagmaLower, then A = L*D*L', where
       L = P(1)*L(1)* ... *P(k)*L(k)* ...,
    i.e., L is a product of terms P(k)*L(k), where k increases from 1 to
    n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
    and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
    defined by IPIV(k), and L(k) is a unit lower triangular matrix, such
    that if the diagonal block D(k) is of order s (s = 1 or 2), then
  
               (   I    0     0   )  k-1
       L(k) =  (   0    I     0   )  s
               (   0    v     I   )  n-k-s+1
                  k-1   s  n-k-s+1
  
    If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
    If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
    and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
 
    @ingroup magma_hetrf
*******************************************************************************/
extern "C" magma_int_t
magma_zhetrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv, magma_int_t *info)
{
    #define  A(i_, j_) ( A + (i_) + (j_)*lda )
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)

    /* .. Local Scalars .. */
    magma_int_t nb = magma_get_zhetrf_nb(n);
    magma_int_t iinfo = 0, nk, kb, j, k;

    /* Test the input parameters. */
    *info = 0;
    bool upper = (uplo == MagmaUpper);
    if ( ! upper && uplo != MagmaLower ) {
        *info = -1;
    } else if ( n < 0 ) {
        *info = -2;
    } else if ( lda < max( 1, n ) ) {
        *info = -4;
    }
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // TODO fix memory leak of dA if dW fails
    magma_int_t ldda = magma_roundup( n, 32 );
    magmaDoubleComplex_ptr dA, dW;
    if ((MAGMA_SUCCESS != magma_zmalloc( &dA, n*ldda  )) ||
        (MAGMA_SUCCESS != magma_zmalloc( &dW, (1+nb)*ldda ))) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    magma_device_t cdev;
    magma_getdevice( &cdev );

    magma_queue_t queues[2];
    magma_event_t event;
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create( &event );
    trace_init( 1, 1, 2, queues );

    /* copy matrix to GPU */
    trace_gpu_start( 0, 0, "set", "setA" );
    //magma_zsetmatrix_async( n, n, A(0,0), lda, dA(0,0), ldda, queues[0] );
    if ( upper ) {
        for (k = 0; k < n; k += nb ) {
            kb = min(nb, n-k);
            magma_zsetmatrix_async( k+kb, kb, A(0,k), lda, dA(0,k), ldda, queues[0] );
        }
    } else {
        for (k = 0; k < n; k += nb ) {
            kb = min(nb, n-k);
            magma_zsetmatrix_async( n-k, kb, A(k,k), lda, dA(k,k), ldda, queues[0] );
        }
    }
    trace_gpu_end( 0, 0 );

    if ( upper ) {
        /* Factorize A as U*D*U' using the upper triangle of A

           K is the main loop index, decreasing from N to 1 in steps of
           KB, where KB is the number of columns factorized by ZLAHEF;
           KB is either NB or NB-1, or K for the last block */

        kb = min(n,nb);
        for (k = n-1; k >= 0; k -= kb ) {
            nk = k+1;
            kb = min(nb, nk);

            if ( k+1 > nb ) {
                /* Factorize columns k-kb+1:k of A and use blocked code to
                   update columns 1:k-kb */

                magma_zlahef_gpu( MagmaUpper, nk, kb, &kb, dA( 0, 0 ), ldda,
                                  &ipiv[0], dW, ldda, queues, &iinfo );

                // copying the panel back to CPU
                magma_event_record( event, queues[0] );
                magma_queue_wait_event( queues[1], event );
                trace_gpu_start( 0, 1, "get", "get" );
                magma_zgetmatrix_async( nk, kb, dA(0,nk-kb), ldda, A(0,nk-kb), lda, queues[1] );  
                trace_gpu_end( 0, 1 );
            } else {
                /* Use unblocked code to factorize columns 1:k of A */

                magma_queue_sync( queues[0] );
                magma_zgetmatrix( nk, nk, dA( 0, 0 ), ldda, A( 0, 0 ), lda, queues[0] );
                lapackf77_zhetf2( MagmaUpperStr, &nk, A( 0, 0 ), &lda, &ipiv[0], &iinfo );
                kb = k+1;
            }

            /* Set INFO on the first occurrence of a zero pivot */

            if ( *info == 0 && iinfo > 0 ) *info = iinfo;
        }
    } else {
        /* Factorize A as L*D*L' using the lower triangle of A

           K is the main loop index, increasing from 1 to N in steps of
           KB, where KB is the number of columns factorized by ZLAHEF;
           KB is either NB or NB-1, or N-K+1 for the last block */

        for (k = 0; k < n; k += kb ) {
            nk = n-k;
            kb = min(nb, n - k);
            if ( k < n-nb ) {
                /* Factorize columns k:k+kb-1 of A and use blocked code to
                   update columns k+kb:n */
                magma_zlahef_gpu( MagmaLower, nk, nb, &kb, dA( k, k ), ldda,
                                  &ipiv[k], dW, ldda, queues, &iinfo );

                // copying the panel back to CPU
                magma_event_record( event, queues[0] );
                magma_queue_wait_event( queues[1], event );
                trace_gpu_start( 0, 1, "get", "get" );
                magma_zgetmatrix_async( nk, kb, dA(k,k), ldda, A(k,k), lda, queues[1] );
                trace_gpu_end( 0, 1 ); 
            }
            else {
                /* Use unblocked code to factorize columns k:n of A */
                magma_queue_sync( queues[0] );
                magma_zgetmatrix( nk, nk, dA(k,k), ldda, A(k,k), lda, queues[0] );
                lapackf77_zhetf2( MagmaLowerStr, &nk, A( k, k ), &lda, &ipiv[k], &iinfo );
            }
            /* Set INFO on the first occurrence of a zero pivot */
            if ( *info == 0 && iinfo > 0 ) *info = iinfo + k;
            /* Adjust IPIV */
            for (j = k; j < k + kb; j ++) {
                if ( ipiv[j] > 0 ) {
                    ipiv[j] = ipiv[j] + k;
                } else {
                    ipiv[j] = ipiv[j] - k;
                }
            }
        }
    }

    trace_finalize( "zhetrf.svg", "trace.css" );
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( event );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_free( dA );
    magma_free( dW );
    
    return *info;
}   /* End of ZHETRF */
