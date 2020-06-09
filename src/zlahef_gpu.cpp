/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ichi Yamazaki
       @author Stan Tomov

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "trace.h"

#define COMPLEX

/***************************************************************************//**
    Purpose
    =======

    ZLAHEF computes a partial factorization of a complex Hermitian
    matrix A using the Bunch-Kaufman diagonal pivoting method. The
    partial factorization has the form:

    A  =  ( I  U12 ) ( A11  0  ) (  I    0   )  if UPLO = MagmaUpper, or:
          ( 0  U22 ) (  0   D  ) ( U12' U22' )

    A  =  ( L11  0 ) (  D   0  ) ( L11' L21' )  if UPLO = MagmaLower
          ( L21  I ) (  0  A22 ) (  0    I   )

    where the order of D is at most NB. The actual order is returned in
    the argument KB, and is either NB or NB-1, or N if N <= NB.
    Note that U' denotes the conjugate transpose of U.

    ZLAHEF is an auxiliary routine called by ZHETRF. It uses blocked code
    (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = MagmaUpper) or
    A22 (if UPLO = MagmaLower).

    Arguments
    ---------
    @param[in]
    uplo    CHARACTER
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is stored:
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nb      INTEGER
            The maximum number of columns of the matrix A that should be
            factored.  NB should be at least 2 to allow for 2-by-2 pivot
            blocks.

    @param[out]
    kb      INTEGER
            The number of columns of A that were actually factored.
            KB is either NB-1 or NB, or N if N <= NB.

    @param[in,out]
    dA      COMPLEX*16 array on GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, A contains details of the partial factorization.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    ipiv    INTEGER array, dimension (N)
            Details of the interchanges and the block structure of D.
            If UPLO = MagmaUpper, only the last KB elements of ipiv are set;
            if UPLO = MagmaLower, only the first KB elements are set.
    \n
            If ipiv(k) > 0, then rows and columns k and ipiv(k) were
            interchanged and D(k,k) is a 1-by-1 diagonal block.
            If UPLO = MagmaUpper and ipiv(k) = ipiv(k-1) < 0, then rows and
            columns k-1 and -ipiv(k) were interchanged and D(k-1:k,k-1:k)
            is a 2-by-2 diagonal block.  If UPLO = MagmaLower and ipiv(k) =
            ipiv(k+1) < 0, then rows and columns k+1 and -ipiv(k) were
            interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
 
    @param[out]
    dW      (workspace) COMPLEX*16 array, dimension (LDDW,NB)
 
    @param[in]
    lddw    INTEGER
            The leading dimension of the array W.  LDDW >= max(1,N).

    @param[in]
    queues  magma_queue_t
            queues contain the queues used for the partial factorization.
            Currently, only one queue is used.

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     > 0: if INFO = k, D(k,k) is exactly zero.  The factorization
                 has been completed, but the block diagonal matrix D is
                 exactly singular.
  
    @ingroup magma_lahef
*******************************************************************************/
extern "C" magma_int_t
magma_zlahef_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDoubleComplex_ptr dW, magma_int_t lddw,
    magma_queue_t queues[],
    magma_int_t *info)
{
    /* .. Parameters .. */
    double d_zero  = 0.0;
    double d_one   = 1.0;
    double d_eight = 8.0;
    double d_seven = 7.0;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t upper = (uplo == MagmaUpper);
    magma_int_t ione = 1;
  
    /* .. Local Scalars .. */
    magma_int_t imax = 0, jmax = 0, kk, kkW, kp, kstep, iinfo;
    double   abs_akk, alpha, colmax, R1, rowmax;
    magmaDoubleComplex Zimax, Z;

    #define dA(i, j)  (dA[(j)*ldda  + (i)])
    #define dW(i, j)  (dW[(j)*lddw  + (i)])

    /* .. Executable Statements .. */
    *info = 0;

    /* Initialize alpha for use in choosing pivot block size. */
    alpha = ( d_one+sqrt( d_seven ) ) / d_eight;

    if ( upper ) {
        /* Factorize the trailing columns of A using the upper triangle
           of A and working backwards, and compute the matrix W = U12*D
           for use in updating A11 (note that conjg(W) is actually stored)

           K is the main loop index, decreasing from N in steps of 1 or 2

           KW is the column of W which corresponds to column K of A */

        magma_int_t k, kw = 0;
        for (k = n-1; k+1 > max(n-nb+1, nb); k -= kstep) {
            kw = nb - (n-k);

            /* Copy column K of A to column KW of W and update it */

            magma_zcopy( k+1, &dA( 0, k ), 1, &dW( 0, kw ), 1, queues[0] );
            // set imaginary part of diagonal to be zero
            #ifdef COMPLEX
            magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( k, kw ))+1, 1, queues[0] );
            #endif
 
            if (k+1 < n) {
                magma_zgemv( MagmaNoTrans, k+1, n-(k+1),
                             c_neg_one, &dA( 0, k+1 ),  ldda,
                                        &dW( k, kw+1 ), lddw,
                             c_one,     &dW( 0, kw ),   ione,
                             queues[0] );

                // set imaginary part of diagonal to be zero
                #ifdef COMPLEX
                magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( k, kw ))+1, 1, queues[0] );
                #endif
            }

            kstep = 1;

            /* Determine rows and columns to be interchanged and whether
               a 1-by-1 or 2-by-2 pivot block will be used */

            magma_zgetvector_async( 1, &dW( k, kw ), 1, &Z, 1, queues[0] );
            magma_queue_sync( queues[0] );
            abs_akk = fabs( MAGMA_Z_REAL( Z ) );

            /* imax is the row-index of the largest off-diagonal element in
               column K, and colmax is its absolute value */

            if ( k > 0 ) {
                // magma is one-base
                imax = magma_izamax( k, &dW( 0, kw ), 1, queues[0] ) - 1;
                magma_zgetvector( 1, &dW( imax, kw ), 1, &Z, 1, queues[0] );
                colmax = MAGMA_Z_ABS1( Z );
            } else {
                colmax = d_zero;
            }

            if ( max( abs_akk, colmax ) == 0.0 ) {
                /* Column K is zero: set INFO and continue */
                if ( *info == 0 ) *info = k;

                kp = k;

                #ifdef COMPLEX
                magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dA( k, k ))+1, 1, queues[0] );
                #endif
            } else {
                if ( abs_akk >= alpha*colmax ) {
                    /* no interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* Copy column imax to column KW-1 of W and update it */
                    magma_zcopy( imax+1, &dA( 0, imax ), 1, &dW( 0, kw-1 ), 1, queues[0] );
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( imax, kw-1 ))+1, 1, queues[0] );
                    #endif

                    #ifdef COMPLEX
                    magmablas_zlacpy_conj( k-imax, &dA( imax, imax+1 ), ldda, &dW( imax+1, kw-1 ), 1, queues[0] );
                    #else
                    magma_zcopy( k-imax, &dA( imax, imax+1 ), ldda, &dW( imax+1, kw-1 ), 1, queues[0] );
                    #endif
                    if ( k+1 < n ) {
                        magma_zgemv( MagmaNoTrans, k+1, n-(k+1),
                                     c_neg_one, &dA( 0, k+1 ),     ldda,
                                                &dW( imax, kw+1 ), lddw,
                                     c_one,     &dW( 0, kw-1 ),    ione,
                                     queues[0] );

                        #ifdef COMPLEX
                        magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( imax, kw-1 ))+1, 1, queues[0] );
                        #endif
                    }
                    magma_zgetvector_async( 1, &dW( imax, kw-1 ), 1, &Zimax, 1, queues[0] );

                    /* jmax is the column-index of the largest off-diagonal
                       element in row imax, and rowmax is its absolute value */

                    jmax = imax + magma_izamax( k-imax, &dW( imax+1, kw-1 ), 1, queues[0] );
                    magma_zgetvector( 1, &dW( jmax, kw-1 ), 1, &Z, 1, queues[0] );
                    rowmax = MAGMA_Z_ABS1( Z );
                    if ( imax > 0 ) {
                        // magma is one-base
                        jmax = magma_izamax( imax, &dW( 0, kw-1 ), 1, queues[0] ) - 1;
                        magma_zgetvector( 1, &dW( jmax, kw-1 ), 1, &Z, 1, queues[0] );
                        rowmax = max( rowmax, MAGMA_Z_ABS1( Z  ) );
                    }

                    if ( abs_akk >= alpha*colmax*( colmax / rowmax ) ) {
                        /* no interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if ( fabs( MAGMA_Z_REAL( Zimax ) ) >= alpha*rowmax ) {
                        /* interchange rows and columns K and imax, use 1-by-1
                           pivot block */
                        kp = imax;

                        /* copy column KW-1 of W to column KW */
                        magma_zcopy( k+1, &dW( 0, kw-1 ), 1, &dW( 0, kw ), 1, queues[0] );
                    } else {
                        /* interchange rows and columns K-1 and imax, use 2-by-2
                           pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }
                kk = k - kstep + 1;
                kkW = nb - (n - kk);

                /* Updated column kp is already stored in column kkW of W */

                if ( kp != kk ) {
                    /* Interchange rows kk and kp in last kk columns of A and W */
                    // note: row-swap A(:,kk)
                    magmablas_zswap( n-kk, &dA( kk, kk ),  ldda, &dA( kp, kk ),  ldda, queues[0] );
                    magmablas_zswap( n-kk, &dW( kk, kkW ), lddw, &dW( kp, kkW ), lddw, queues[0] );

                    /* Copy non-updated column kk to column kp */
                    #ifdef COMPLEX
                    magmablas_zlacpy_conj( kk-kp-1, &dA( kp+1, kk ), 1, &dA( kp, kp+1 ), ldda, queues[0] );
                    #else
                    magma_zcopy( kk-kp-1, &dA( kp+1, kk ), 1, &dA( kp, kp+1 ), ldda, queues[0] );
                    #endif

                    // now A(kp,kk) should be A(kk,kk), and copy to A(kp,kp)
                    magma_zcopy( kp+1, &dA( 0, kk ), 1, &dA( 0, kp ), 1, queues[0] );
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dA( kp, kp ))+1, 1, queues[0] );
                    #endif
                }

                if ( kstep == 1 ) {
                    /* 1-by-1 pivot block D(k): column KW of W now holds
                       W(k) = U(k)*D(k)
                       where U(k) is the k-th column of U
                       Store U(k) in column k of A */
                    magma_zcopy( k+1, &dW( 0, kw ), 1, &dA( 0, k ), 1, queues[0] );
                    if ( k > 0 ) {
                        magma_zgetvector_async( 1, &dA( k, k ), 1, &Z, 1, queues[0] );
                        magma_queue_sync( queues[0] );
                        R1 = d_one / MAGMA_Z_REAL( Z );
                        magma_zdscal( k, R1, &dA( 0, k ), 1, queues[0] );

                        /* Conjugate W(k) */

                        #ifdef COMPLEX
                        magmablas_zlacpy_conj( k, &dW( 0, kw ), 1, &dW( 0, kw ), 1, queues[0] );
                        #endif
                    }
                } else {
                    /* 2-by-2 pivot block D(k): columns KW and KW-1 of W now
                       hold
                       ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
                       where U(k) and U(k-1) are the k-th and (k-1)-th columns
                       of U */
                    if ( k > 1 ) {
                        /* Store U(k) and U(k-1) in columns k and k-1 of A */
                        magmablas_zlascl_2x2( MagmaUpper, k-1, &dW(0, kw-1), lddw, &dA(0,k-1), ldda, queues[0], &iinfo );
                    }

                    /* Copy D(k) to A */
                    magma_zcopymatrix( 2, 2, &dW( k-1, kw-1 ), lddw, &dA( k-1, k-1 ), ldda, queues[0] );

                    /* Conjugate W(k) and W(k-1) */

                    #ifdef COMPLEX
                    magmablas_zlacpy_conj( k,   &dW( 0, kw ),   1, &dW( 0, kw ),   1, queues[0] );
                    magmablas_zlacpy_conj( k-1, &dW( 0, kw-1 ), 1, &dW( 0, kw-1 ), 1, queues[0] );
                    #endif
                }
            }

            /* Store details of the interchanges in ipiv */

            if ( kstep == 1 ) {
                ipiv[ k ] = 1+kp;
            } else {
                ipiv[ k ] = -(1+kp);
                ipiv[ k-1 ] = -(1+kp);
            }
        }

        /* Update the upper triangle of A11 (= A(1:k,1:k)) as

           A11 := A11 - U12*D*U12' = A11 - U12*W'

           computing blocks of NB columns at a time (note that conjg(W) is
           actually stored) */

        kw = nb - (n-k);
        for (magma_int_t j = ( k / nb )*nb; j >= 0; j -= nb ) {
            magma_int_t jb = min( nb, k-j+1 );

            #ifdef SYMMETRIC_UPDATE
                /* Update the upper triangle of the diagonal block */
                for (magma_int_t jj = j; jj < j + jb; jj++) {
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dA( jj, jj ))+1, 1, queues[0] );
                    #endif
                    magma_zgemv( MagmaNoTrans, jj-j+1, n-(k+1),
                                 c_neg_one, &dA( j, k+1 ),   ldda,
                                            &dW( jj, kw+1 ), lddw,
                                 c_one,     &dA( j, jj ),    1,
                                 queues[0] );
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dA( jj, jj ))+1, 1, queues[0] );
                    #endif
                }
    
                /* Update the rectangular superdiagonal block */
                magma_zgemm( MagmaNoTrans, MagmaTrans, j, jb, n-(k+1),
                             c_neg_one, &dA( 0, k+1 ),  ldda,
                                        &dW( j, kw+1 ), lddw,
                             c_one,     &dA( 0, j ),    ldda,
                             queues[0] );
            #else
                #ifdef COMPLEX
                magmablas_dlaset(MagmaFull, 1, jb, d_zero, d_zero, ((magmaDouble_ptr)&dA( j, j ))+1, 2*(1+ldda), queues[0] );
                #endif
                magma_zgemm( MagmaNoTrans, MagmaTrans, j+jb, jb, n-(k+1),
                             c_neg_one, &dA( 0, k+1 ),  ldda,
                                        &dW( j, kw+1 ), lddw,
                             c_one,     &dA( 0, j ),    ldda,
                             queues[0] );
                #ifdef COMPLEX
                magmablas_dlaset(MagmaFull, 1, jb, d_zero, d_zero, ((magmaDouble_ptr)&dA( j, j ))+1, 2*(1+ldda), queues[0] );
                #endif
            #endif
        }

        /* Put U12 in standard form by partially undoing the interchanges
           in columns k+1:n */

        for (magma_int_t j = k+1; j < n; ) {
            magma_int_t jj = j;
            magma_int_t jp = ipiv[ j ];
            if ( jp < 0 ) {
                jp = -jp;
                j = j + 1;
            }
            j = j + 1;
            jp = jp - 1;
            if ( jp != jj && j < n )
                magmablas_zswap( n-j, &dA( jp, j ), ldda, &dA( jj, j ), ldda, queues[0] );
        }

        /* Set KB to the number of columns factorized */
        *kb = n - (k+1);
    }
    else {
        /* Factorize the leading columns of A using the lower triangle
           of A and working forwards, and compute the matrix W = L21*D
           for use in updating A22 (note that conjg(W) is actually stored)

           K is the main loop index, increasing from 1 in steps of 1 or 2 */

        magma_int_t k;
        for (k = 0; k < min(nb-1,n); k += kstep) {
            /* Copy column K of A to column K of W and update it */

            /* -------------------------------------------- */
            trace_gpu_start( 0, 0, "copy", "copyAk" );
            magma_zcopy( n-k, &dA( k, k ), 1, &dW( k, k ), 1, queues[0] );

            // set imaginary part of diagonal to be zero
            #ifdef COMPLEX
            magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( k, k ))+1, 1, queues[0] );
            #endif
            trace_gpu_end( 0, 0 );
            /* -------------------------------------------- */

            trace_gpu_start( 0, 0, "gemv", "gemv" );
            magma_zgemv( MagmaNoTrans, n-k, k,
                         c_neg_one, &dA( k, 0 ), ldda,
                                    &dW( k, 0 ), lddw,
                         c_one,     &dW( k, k ), ione,
                         queues[0] );
            // re-set imaginary part of diagonal to be zero
            #ifdef COMPLEX
            magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( k, k ))+1, 1, queues[0] );
            #endif
            trace_gpu_end( 0, 0 );

            kstep = 1;

            /* Determine rows and columns to be interchanged and whether
               a 1-by-1 or 2-by-2 pivot block will be used */

            magma_zgetvector_async( 1, &dW( k, k ), 1, &Z, 1, queues[0] );
            magma_queue_sync( queues[0] );
            abs_akk = fabs( MAGMA_Z_REAL( Z ) );

            /* imax is the row-index of the largest off-diagonal element in
               column K, and colmax is its absolute value */

            if ( k < n-1 ) {
                // magmablas is one-base
                trace_gpu_start( 0, 0, "max", "max" );
                imax = k + magma_izamax( n-k-1, &dW(k+1,k), 1, queues[0] );
                trace_gpu_end( 0, 0 );
                magma_zgetvector( 1, &dW( imax, k ), 1, &Z, 1, queues[0] );
                colmax = MAGMA_Z_ABS1( Z );
            }
            else {
                colmax = d_zero;
            }

            if ( max( abs_akk, colmax ) == 0.0 ) {
                /* Column K is zero: set INFO and continue */

                if ( *info == 0 ) *info = k;
                kp = k;

                // make sure the imaginary part of diagonal is zero
                #ifdef COMPLEX
                magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dA( k, k ))+1, 1, queues[0] );
                #endif
            } else {
                if ( abs_akk >= alpha*colmax ) {
                    /* no interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* Copy column imax to column K+1 of W and update it */
                    trace_gpu_start( 0, 0, "copy", "copy" );
                    #ifdef COMPLEX
                    magmablas_zlacpy_conj( imax-k, &dA( imax, k ), ldda, &dW( k, k+1 ), 1, queues[0] );
                    #else
                    magma_zcopy( imax-k, &dA( imax, k ), ldda, &dW( k, k+1 ), 1, queues[0] );
                    #endif

                    magma_zcopy( n-imax, &dA( imax, imax ), 1, &dW( imax, k+1 ), 1, queues[0] );
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( imax, k+1 ))+1, 1, queues[0] );
                    #endif
                    trace_gpu_end( 0, 0 );

                    trace_gpu_start( 0, 0, "gemv", "gemv" );
                    magma_zgemv( MagmaNoTrans, n-k, k,
                                 c_neg_one, &dA( k, 0 ),    ldda,
                                            &dW( imax, 0 ), lddw,
                                 c_one,     &dW( k, k+1 ),  ione,
                                 queues[0] );
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dW( imax, k+1 ))+1, 1, queues[0] );
                    #endif
                    trace_gpu_end( 0, 0 );

                    magma_zgetvector_async( 1, &dW( imax, k+1 ), 1, &Zimax, 1, queues[0] );

                    /* jmax is the column-index of the largest off-diagonal
                       element in row imax, and rowmax is its absolute value */

                    // magmablas is one-base
                    trace_gpu_start( 0, 0, "max", "max" );
                    jmax = k-1 + magma_izamax( imax-k, &dW(k, k+1), 1, queues[0] );
                    trace_gpu_end( 0, 0 );
                    magma_zgetvector( 1, &dW( jmax, k+1 ), 1, &Z, 1, queues[0] );
                    rowmax = MAGMA_Z_ABS1( Z );
                    if ( imax < n-1 ) {
                        // magmablas is one-base
                        trace_gpu_start( 0, 0, "max", "max" );
                        jmax = imax + magma_izamax( (n-1)-imax, &dW( imax+1, k+1 ), 1, queues[0] );
                        trace_gpu_end( 0, 0 );
                        magma_zgetvector( 1, &dW( jmax, k+1 ), 1, &Z, 1, queues[0] );
                        rowmax = max( rowmax, MAGMA_Z_ABS1( Z ) );
                    }

                    if ( abs_akk >= alpha*colmax*( colmax / rowmax ) ) {
                        /* no interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if ( fabs( MAGMA_Z_REAL( Zimax ) ) >= alpha*rowmax ) {
                        /* interchange rows and columns K and imax, use 1-by-1
                           pivot block */
                        kp = imax;

                        /* copy column K+1 of W to column K */
                        trace_gpu_start( 0, 0, "copy", "copy" );
                        magma_zcopy( n-k, &dW( k, k+1 ), 1, &dW( k, k ), 1, queues[0] );
                        trace_gpu_end( 0, 0 );
                    } else {
                        /* interchange rows and columns K+1 and imax, use 2-by-2
                           pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                kk = k + kstep - 1;

                /* Updated column kp is already stored in column kk of W */

                if ( kp != kk ) {
                    /* Copy non-updated column kk to column kp */

                    /* -------------------------------------------- */
                    trace_gpu_start( 0, 0, "copy", "copy" );
                    #ifdef COMPLEX
                    magmablas_zlacpy_conj( kp-kk, &dA( kk, kk ), 1, &dA( kp, kk ), ldda, queues[0] );
                    #else
                    magma_zcopy( kp-kk, &dA( kk, kk ), 1, &dA( kp, kk ), ldda, queues[0] );
                    #endif
                    magma_zcopy( n-kp, &dA( kp, kk ), 1, &dA( kp, kp ), 1, queues[0] );
                    #ifdef COMPLEX
                    magma_dsetvector_async( 1, &d_zero, 1, ((magmaDouble_ptr)&dA( kp, kp ))+1, 1, queues[0] );
                    #endif
                    trace_gpu_end( 0, 0 );
                    /* -------------------------------------------- */

                    /* Interchange rows kk and kp in first kk columns of A and W */

                    trace_gpu_start( 0, 0, "permute", "swap-backward" );
                    magmablas_zswap( kk+1, &dA( kk, 0 ), ldda, &dA( kp, 0 ), ldda, queues[0] );
                    magmablas_zswap( kk+1, &dW( kk, 0 ), lddw, &dW( kp, 0 ), lddw, queues[0] );
                    trace_gpu_end( 0, 0 );
                }

                if ( kstep == 1 ) {
                    /* 1-by-1 pivot block D(k): column k of W now holds
                       W(k) = L(k)*D(k)
                       where L(k) is the k-th column of L
                       Store L(k) in column k of A */
                    trace_gpu_start( 0, 0, "copy", "copy" );
                    magma_zcopy( n-k, &dW( k, k ), 1, &dA( k, k ), 1, queues[0] );
                    trace_gpu_end( 0, 0 );

                    if ( k < n-1 ) {
                        magma_zgetvector_async( 1, &dA( k, k ), 1, &Z, 1, queues[0] );
                        R1 = d_one / MAGMA_Z_REAL( Z );
                        magma_queue_sync( queues[0] );
                        trace_gpu_start( 0, 0, "scal", "scal-1" );
                        magma_zdscal((n-1)-k, R1, &dA( k+1, k ), 1, queues[0]);
                        trace_gpu_end( 0, 0 );

                        /* Conjugate W(k) */
                        #ifdef COMPLEX
                        magmablas_zlacpy_conj( (n-1)-k, &dW( k+1, k ), 1, &dW( k+1, k ), 1, queues[0] );
                        #endif
                    }
                } else {
                    /* 2-by-2 pivot block D(k): columns k and k+1 of W now hold
                     ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
                    where L(k) and L(k+1) are the k-th and (k+1)-th columns
                    of L */
                    trace_gpu_start( 0, 0, "scal", "scal-2" );
                    if (n > k+2)
                        magmablas_zlascl_2x2( MagmaLower, n-(k+2), &dW(k,k), lddw, &dA(k+2,k), ldda, queues[0], &iinfo );

                    /* Copy D(k) to A */
                    magma_zcopymatrix( 2, 2, &dW( k, k ), lddw, &dA( k, k ), ldda, queues[0] );

                    /* Conjugate W(k) and W(k+1) */
                    #ifdef COMPLEX
                    magmablas_zlacpy_conj( (n-1)-k,   &dW( k+1, k ),   1, &dW( k+1, k ),   1, queues[0] );
                    magmablas_zlacpy_conj( (n-1)-k-1, &dW( k+2, k+1 ), 1, &dW( k+2, k+1 ), 1, queues[0] );
                    #endif
                    trace_gpu_end( 0, 0 );
                }
            }

            /* Store details of the interchanges in ipiv */

            if ( kstep == 1 ) {
                ipiv[k] = kp+1;
            } else {
                ipiv[k] = -kp-1;
                ipiv[k+1] = -kp-1;
            }
        }

        /* Update the lower triangle of A22 (= A(k:n,k:n)) as

           A22 := A22 - L21*D*L21' = A22 - L21*W'

           computing blocks of NB columns at a time (note that conjg(W) is
           actually stored) */

        for( magma_int_t j = k; j < n; j += nb ) {
            magma_int_t jb = min( nb, n-j );

            /* Update the lower triangle of the diagonal block */

            trace_gpu_start( 0, 0, "gemm", "gemm" );
            #ifdef SYMMETRIC_UPDATE
                for (magma_int_t jj = j; jj < j + jb; jj++) {
                    magma_int_t jnb = j + jb - jj;
    
                    /* -------------------------------------------- */
                    magma_zgemv( MagmaNoTrans, jnb, k,
                                 c_neg_one, &dA( jj, 0 ),  ldda,
                                            &dW( jj, 0 ),  lddw,
                                 c_one,     &dA( jj, jj ), ione,
                                 queues[0] );
                    /* -------------------------------------------- */
                }

                /* Update the rectangular subdiagonal block */
                if ( j+jb < n ) {
                    magma_int_t nk = n - (j+jb);
    
                    /* -------------------------------------------- */
                    magma_zgemm( MagmaNoTrans, MagmaTrans, nk, jb, k,
                                 c_neg_one, &dA( j+jb, 0 ), ldda,
                                            &dW( j, 0 ),    lddw,
                                 c_one,     &dA( j+jb, j ), ldda,
                                 queues[0] );
                    /* ------------------------------------------- */
                }
            #else
                #ifdef COMPLEX
                magmablas_dlaset(MagmaFull, 1, jb, d_zero, d_zero, ((magmaDouble_ptr)&dA( j, j ))+1, 2*(1+ldda), queues[0] );
                #endif
                magma_zgemm( MagmaNoTrans, MagmaTrans, n-j, jb, k,
                             c_neg_one, &dA( j, 0 ), ldda,
                                        &dW( j, 0 ), lddw,
                             c_one,     &dA( j, j ), ldda,
                             queues[0] );
                #ifdef COMPLEX
                magmablas_dlaset(MagmaFull, 1, jb, d_zero, d_zero, ((magmaDouble_ptr)&dA( j, j ))+1, 2*(1+ldda), queues[0] );
                #endif
            #endif
            trace_gpu_end( 0, 0 );
        }

        /* Put L21 in standard form by partially undoing the interchanges
           in columns 1:k-1 */

        for (magma_int_t j = k; j > 0; ) {
            magma_int_t jj = j;
            magma_int_t jp = ipiv[j-1];
            if ( jp < 0 ) {
                jp = -jp;
                j--;
            }
            j--;
            if ( jp != jj && j >= 1 ) {
                trace_gpu_start( 0, 0, "permute", "perm" );
                magmablas_zswap( j, &dA( jp-1, 0 ), ldda, &dA( jj-1, 0 ), ldda, queues[0] );
                trace_gpu_end( 0, 0 );
                magma_queue_sync( queues[0] );
            }
        }

        /* Set KB to the number of columns factorized */
        *kb = k;
    }

    return *info;
    /* End of ZLAHEF */
}
