/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define FAST_HEMV

/***************************************************************************//**
    Purpose
    -------
    ZHETRD reduces a complex Hermitian matrix A to real symmetric
    tridiagonal form T by an orthogonal similarity transformation:
    Q**H * A * Q = T.

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
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = MagmaUpper, the diagonal and first superdiagonal
            of A are overwritten by the corresponding elements of the
            tridiagonal matrix T, and the elements above the first
            superdiagonal, with the array TAU, represent the orthogonal
            matrix Q as a product of elementary reflectors; if UPLO
            = MagmaLower, the diagonal and first subdiagonal of A are over-
            written by the corresponding elements of the tridiagonal
            matrix T, and the elements below the first subdiagonal, with
            the array TAU, represent the orthogonal matrix Q as a product
            of elementary reflectors. See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    d       COMPLEX_16 array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    @param[out]
    e       COMPLEX_16 array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = MagmaUpper, E(i) = A(i+1,i) if UPLO = MagmaLower.

    @param[out]
    tau     COMPLEX_16 array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= N*NB, where NB is the
            optimal blocksize given by magma_get_zhetrd_nb().
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    If UPLO = MagmaUpper, the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n-1) . . . H(2) H(1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
    A(1:i-1,i+1), and tau in TAU(i).

    If UPLO = MagmaLower, the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(n-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
    and tau in TAU(i).

    The contents of A on exit are illustrated by the following examples
    with n = 5:

    if UPLO = MagmaUpper:                if UPLO = MagmaLower:

      (  d   e   v2  v3  v4 )              (  d                  )
      (      d   e   v3  v4 )              (  e   d              )
      (          d   e   v4 )              (  v1  e   d          )
      (              d   e  )              (  v1  v2  e   d      )
      (                  d  )              (  v1  v2  v3  e   d  )

    where d and e denote diagonal and off-diagonal elements of T, and vi
    denotes an element of the vector defining H(i).

    @ingroup magma_hetrd
*******************************************************************************/
extern "C" magma_int_t
magma_zhetrd(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info)
{
    #define  A(i_, j_) ( A + (i_) + (j_)*lda )
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)

    /* Constants */
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const double             d_one     = MAGMA_D_ONE;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );

    magma_int_t ldda = magma_roundup( n, 32 );
    magma_int_t nb = magma_get_zhetrd_nb( n );

    magma_int_t kk, nx;
    magma_int_t i, j, i_n;
    magma_int_t iinfo;
    magma_int_t ldw, lddw, lwkopt;
    magma_int_t lquery;

    *info = 0;
    bool upper = (uplo == MagmaUpper);
    lquery = (lwork == -1);
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    } else if (lwork < nb*n && ! lquery) {
        *info = -9;
    }

    /* Determine the block size. */
    ldw = n;
    lddw = ldda;
    lwkopt = n * nb;
    if (*info == 0) {
        work[0] = magma_zmake_lwork( lwkopt );
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    /* Quick return if possible */
    if (n == 0) {
        work[0] = c_one;
        return *info;
    }

    magmaDoubleComplex *work2;
    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work2, n )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magmaDoubleComplex *dA;
    #ifdef FAST_HEMV
    magma_int_t ldwork2 = ldda*magma_ceildiv(n,64);
    #else
    magma_int_t ldwork2 = 0;
    #endif
    if (MAGMA_SUCCESS != magma_zmalloc( &dA, ldda*n + 2*lddw*nb + ldwork2 )) {
        magma_free_cpu( work2 );
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magmaDoubleComplex *dwork  = dA + ldda*n;
    #ifdef FAST_HEMV
    magmaDoubleComplex *dwork2 = dwork + 2*lddw*nb;
    #endif

    #ifdef MAGMA_DISABLE_MKL_THREADING_ISSUE_BLAS1
    magma_int_t lapack_nthread = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(lapack_nthread > 1 ? 2 : 1);
    #endif

    // nx <= n is required
    // use LAPACK for n < 3000, otherwise switch at 512
    if (n < 3000)
        nx = n;
    else
        nx = 512;
    
    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    // clear out dwork in case it has NANs (used as y in zhemv)
    // rest of dwork (used as work in magmablas_zhemv) doesn't need to be cleared
    magmablas_zlaset( MagmaFull, n, nb, c_zero, c_zero, dwork, lddw, queue );

    if (upper) {
        /* Copy the matrix to the GPU */
        magma_zsetmatrix( n, n, A(0, 0), lda, dA(0, 0), ldda, queue );

        /* Reduce the upper triangle of A.
           Columns 1:kk are handled by the unblocked method. */
        kk = n - magma_roundup( n - nx, nb );

        for (i = n - nb; i >= kk; i -= nb) {
            /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */
            
            /* Get the current panel (no need for the 1st iteration) */
            if (i != n-nb)
                magma_zgetmatrix( i+nb, nb, dA(0, i), ldda, A(0, i), lda, queue );
            
            #ifdef FAST_HEMV
            magma_zlatrd2( uplo, i+nb, nb, A(0, 0), lda, e, tau,
                           work, ldw, work2, n, dA(0, 0), ldda, dwork, lddw,
                           dwork2, ldwork2, queue );
            #else
            magma_zlatrd(  uplo, i+nb, nb, A(0, 0), lda, e, tau,
                           work, ldw, work2, n, dA(0, 0), ldda, dwork, lddw, queue );
            #endif

            /* Update the unreduced submatrix A(0:i-2,0:i-2), using an
               update of the form:  A := A - V*W' - W*V' */
            magma_zsetmatrix( i + nb, nb, work, ldw, dwork, lddw, queue );

            magma_zher2k( uplo, MagmaNoTrans, i, nb, c_neg_one,
                          dA(0, i), ldda, dwork, lddw,
                          d_one, dA(0, 0), ldda, queue );
            
            /* Copy superdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i+nb; ++j) {
                *A(j-1,j) = MAGMA_Z_MAKE( e[j - 1], 0 );
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }
        }
        
        magma_zgetmatrix( kk, kk, dA(0, 0), ldda, A(0, 0), lda, queue );
        
        /* Use CPU code to reduce the last or only block */
        lapackf77_zhetrd( uplo_, &kk, A(0, 0), &lda, d, e, tau, work, &lwork, &iinfo );
    }
    else {
        /* Copy the matrix to the GPU */
        if (1 <= n-nx)
            magma_zsetmatrix( n, n, A(0,0), lda, dA(0,0), ldda, queue );

        /* Reduce the lower triangle of A */
        for (i = 0; i < n-nx; i += nb) {
            /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */

            /* Get the current panel (no need for the 1st iteration) */
            if (i != 0)
                magma_zgetmatrix( n-i, nb, dA(i, i), ldda, A(i, i), lda, queue );
            
            #ifdef FAST_HEMV
            magma_zlatrd2( uplo, n-i, nb, A(i, i), lda, &e[i], &tau[i],
                           work, ldw, work2, n, dA(i, i), ldda, dwork, lddw,
                           dwork2, ldwork2, queue );
            #else
            magma_zlatrd(  uplo, n-i, nb, A(i, i), lda, &e[i], &tau[i],
                           work, ldw, work2, n, dA(i, i), ldda, dwork, lddw, queue );
            #endif
            
            /* Update the unreduced submatrix A(i+ib:n,i+ib:n), using
               an update of the form:  A := A - V*W' - W*V' */
            magma_zsetmatrix( n-i, nb, work, ldw, dwork, lddw, queue );

            magma_zher2k( MagmaLower, MagmaNoTrans, n-i-nb, nb, c_neg_one,
                          dA(i+nb, i), ldda, &dwork[nb], lddw,
                          d_one, dA(i+nb, i+nb), ldda, queue );
            
            /* Copy subdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i+nb; ++j) {
                *A(j+1,j) = MAGMA_Z_MAKE( e[j], 0 );
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }
        }

        /* Use CPU code to reduce the last or only block */
        if (1 <= n-nx)
            magma_zgetmatrix( n-i, n-i, dA(i, i), ldda, A(i, i), lda, queue );
        
        i_n = n-i;
        lapackf77_zhetrd( uplo_, &i_n, A(i, i), &lda, &d[i], &e[i],
                          &tau[i], work, &lwork, &iinfo );
    }
    
    magma_free_cpu( work2 );
    magma_free( dA );
    magma_queue_destroy( queue );
    
    work[0] = magma_zmake_lwork( lwkopt );

    #ifdef MAGMA_DISABLE_MKL_THREADING_ISSUE_BLAS1
    magma_set_lapack_numthreads(lapack_nthread);
    #endif
    return *info;
} /* magma_zhetrd */
