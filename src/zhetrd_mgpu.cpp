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
#include "trace.h"

/***************************************************************************//**
    Purpose
    -------
    ZHETRD reduces a complex Hermitian matrix A to real symmetric
    tridiagonal form T by an orthogonal similarity transformation:
    Q**H * A * Q = T.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    nqueue  INTEGER
            The number of GPU queues used for update.  10 >= nqueue > 0.

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
magma_zhetrd_mgpu(
    magma_int_t ngpu,
    magma_int_t nqueue, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info)
{
#define  A(i, j)     (A           + (j)*lda  + (i))
#define dA(id, i, j) (dA[(id)]    + (j)*ldda + (i))
#define dW(id, i, j) (dW[(id)] + (j)*ldda + (i))

    /* Constants */
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const double             d_one     = MAGMA_D_ONE;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    
    magma_int_t nlocal, ldda;
    magma_int_t nb = magma_get_zhetrd_nb(n), ib, ib2;

    #ifdef PROFILE_SY2RK
    double mv_time = 0.0;
    double up_time = 0.0;
    #endif

    magma_int_t kk, nx;
    magma_int_t i, ii, iii, j, dev, i_n;
    magma_int_t iinfo;
    magma_int_t ldwork, lddw, lwkopt, ldwork2, lhwork;
    
    // set pointers to NULL so it is safe to goto CLEANUP if any malloc fails.
    magma_queue_t queues[MagmaMaxGPUs][10] = { { NULL, NULL } };
    magma_queue_t queues0[MagmaMaxGPUs]    = { NULL };
    magmaDoubleComplex *hwork = NULL;
    magmaDoubleComplex_ptr dwork2[MagmaMaxGPUs] = { NULL };
    magmaDoubleComplex_ptr dA[MagmaMaxGPUs]     = { NULL };
    magmaDoubleComplex_ptr dW[MagmaMaxGPUs]     = { NULL };

    *info = 0;
    bool upper = (uplo == MagmaUpper);
    bool lquery = (lwork == -1);
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    } else if (lwork < nb*n && ! lquery) {
        *info = -9;
    } else if ( nqueue > 2 ) {
        *info = 2;  // TODO fix
    }

    /* Determine the block size. */
    ldwork = n;
    lwkopt = n * nb;
    if (*info == 0) {
        work[0] = magma_zmake_lwork( lwkopt );
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (n == 0) {
        work[0] = c_one;
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    //#define PROFILE_SY2RK
    #ifdef PROFILE_SY2RK
    double times[11] = { 0 };
    magma_event_t start, stop;
    float etime;
    magma_setdevice( 0 );
    magma_event_create( &start );
    magma_event_create( &stop  );
    #endif

    ldda = magma_roundup( lda, 32 );
    lddw = ldda;
    nlocal = nb*(1 + n/(nb*ngpu));
    ldwork2 = ldda*( magma_ceildiv( n, nb ) + 1);  // i.e., ldda*(blocks + 1)
    for( dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        // TODO fix memory leak
        if ( MAGMA_SUCCESS != magma_zmalloc( &dA[dev],     nlocal*ldda + 3*lddw*nb ) ||
             MAGMA_SUCCESS != magma_zmalloc( &dwork2[dev], ldwork2 ) ) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto CLEANUP;
        }
        dW[dev] = dA[dev] + nlocal*ldda;
        
        for( kk=0; kk < nqueue; kk++ ) {
            magma_device_t cdev;
            magma_getdevice( &cdev );
            magma_queue_create( cdev, &queues[dev][kk] );
        }
        queues0[dev] = queues[dev][0];
    }
    
    lhwork = nqueue*ngpu*n;
    if ( MAGMA_SUCCESS != magma_zmalloc_pinned( &hwork, lhwork ) ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto CLEANUP;
    }

    // nx <= n is required
    // use LAPACK for n < 3000, otherwise switch at 512
    if (n < 3000)
        nx = n;
    else
        nx = 512;

    if (upper) {
        /* Copy the matrix to the GPU */
        if (1 <= n-nx) {
            magma_zhtodhe( ngpu, uplo, n, nb, A, lda, dA, ldda, queues, &iinfo );
        }

        /*  Reduce the upper triangle of A.
            Columns 1:kk are handled by the unblocked method. */
        for (i = nb*((n-1)/nb); i >= nx; i -= nb) {
            ib = min(nb, n-i);

            ii  = nb*(i/(nb*ngpu));
            dev = (i/nb)%ngpu;

            /* wait for the next panel */
            if (i != nb*((n-1)/nb)) {
                magma_setdevice( dev );
                magma_queue_sync( queues[dev][0] );
            }

            magma_zlatrd_mgpu( ngpu, uplo, i+ib, ib, nb,
                               A(0, 0), lda, e, tau,
                               work, ldwork,
                               dA, ldda, 0,
                               dW, i+ib,
                               hwork,  lhwork,
                               dwork2, ldwork2,
                               queues0 );

            magma_zher2k_mgpu( ngpu, MagmaUpper, MagmaNoTrans, nb, i, ib,
                               c_neg_one, dW, i+ib, 0,
                               d_one,     dA, ldda, 0,
                               nqueue, queues );

            /* get the next panel */
            if (i-nb >= nx ) {
                ib2 = min(nb, n-(i-nb));
                
                ii  = nb*((i-nb)/(nb*ngpu));
                dev = ((i-nb)/nb)%ngpu;
                magma_setdevice( dev );
                
                magma_zgetmatrix_async( (i-nb)+ib2, ib2,
                                        dA(dev, 0, ii), ldda,
                                        A(0, i-nb),     lda,
                                        queues[dev][0] );
            }

            /* Copy superdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i+ib; ++j) {
                if ( j > 0 ) {
                    *A(j-1,j) = MAGMA_Z_MAKE( e[j - 1], 0 );
                }
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }
        } /* end of for i=... */
      
        if ( nx > 0 ) {
            if (1 <= n-nx) { /* else A is already on CPU */
                for (i=0; i < nx; i += nb) {
                    ib = min(nb, n-i);
                    ii  = nb*(i/(nb*ngpu));
                    dev = (i/nb)%ngpu;
                
                    magma_setdevice( dev );
                    magma_zgetmatrix_async( nx, ib,
                                            dA(dev, 0, ii), ldda,
                                            A(0, i),        lda,
                                            queues[dev][0] );
                }
            }
            
            for( dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_queue_sync( queues[dev][0] );
            }
            /* Use CPU code to reduce the last or only block */
            lapackf77_zhetrd( uplo_, &nx, A(0, 0), &lda, d, e, tau,
                              work, &lwork, &iinfo );
        }
    }
    else {
        trace_init( 1, ngpu, nqueue, (magma_queue_t *)queues );
        /* Copy the matrix to the GPU */
        if (1 <= n-nx) {
            magma_zhtodhe( ngpu, uplo, n, nb, A, lda, dA, ldda, queues, &iinfo );
        }

        /* Reduce the lower triangle of A */
        for (i = 0; i < n-nx; i += nb) {
            ib = min(nb, n-i);

            ii  = nb*(i/(nb*ngpu));
            dev = (i/nb)%ngpu;
            /* Reduce columns i:i+ib-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */

            /*   Get the current panel (no need for the 1st iteration) */
            if (i != 0) {
                magma_setdevice( dev );
                trace_gpu_start( dev, 0, "comm", "get" );
                magma_zgetmatrix_async( n-i, ib,
                                        dA(dev, i, ii), ldda,
                                        A(i,i),         lda,
                                        queues[dev][0] );
                trace_gpu_end( dev, 0 );
                magma_queue_sync( queues[dev][0] );
                magma_setdevice( 0 );
            }
            
            magma_zlatrd_mgpu( ngpu, uplo, n-i, ib, nb,
                               A(i, i), lda, &e[i], &tau[i],
                               work, ldwork,
                               dA, ldda, i,
                               dW, n-i,
                               hwork,  lhwork,
                               dwork2, ldwork2,
                               queues0 );

            #ifdef PROFILE_SY2RK
            magma_setdevice( 0 );
            if ( i > 0 ) {
                cudaEventElapsedTime( &etime, start, stop );
                up_time += (etime/1000.0);
            }
            magma_event_record( start, 0 );
            #endif
            
            magma_zher2k_mgpu( ngpu, MagmaLower, MagmaNoTrans, nb, n-i-ib, ib,
                               c_neg_one, dW, n-i, ib,
                               d_one, dA, ldda, i+ib, nqueue, queues );
            
            #ifdef PROFILE_SY2RK
            magma_setdevice( 0 );
            magma_event_record( stop, 0 );
            #endif

            /* Copy subdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i+ib; ++j) {
                if ( j+1 < n ) {
                    *A(j+1,j) = MAGMA_Z_MAKE( e[j], 0 );
                }
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }
        } /* for i=... */

        /* Use CPU code to reduce the last or only block */
        if ( i < n ) {
            iii = i;
            i_n = n-i;
            if ( i > 0 ) {
                for (; i < n; i += nb) {
                    ib = min(nb, n-i);
                    ii  = nb*(i/(nb*ngpu));
                    dev = (i/nb)%ngpu;
                
                    magma_setdevice( dev );
                    magma_zgetmatrix_async( i_n, ib,
                                            dA(dev, iii, ii), ldda,
                                            A(iii, i),        lda,
                                            queues[dev][0] );
                }
                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    magma_queue_sync( queues[dev][0] );
                }
            }
            lapackf77_zhetrd( uplo_, &i_n, A(iii, iii), &lda, &d[iii], &e[iii],
                              &tau[iii], work, &lwork, &iinfo );
        }
    }
    
    for( dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        for( kk=0; kk < nqueue; kk++ ) {
            magma_queue_sync( queues[dev][kk] );
        }
    }
    
    #ifdef PROFILE_SY2RK
    magma_setdevice( 0 );
    if ( n > nx ) {
        cudaEventElapsedTime( &etime, start, stop );
        up_time += (etime/1000.0);
    }
    magma_event_destroy( start );
    magma_event_destroy( stop  );
    #endif

    trace_finalize( "zhetrd.svg", "trace.css" );
    
    #ifdef PROFILE_SY2RK
    printf( " n=%lld nb=%lld\n", (long long) n, (long long) nb );
    printf( " Time in ZLARFG: %.2e seconds\n", times[0] );
    //printf( " Time in ZHEMV : %.2e seconds\n", mv_time );
    printf( " Time in ZHER2K: %.2e seconds\n", up_time );
    #endif
    
CLEANUP:
    for( dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        for( kk=0; kk < nqueue; kk++ ) {
            magma_queue_destroy( queues[dev][kk] );
        }
        magma_free( dA[dev] );
        magma_free( dwork2[dev] );
    }
    magma_free_pinned( hwork );
    
    magma_setdevice( orig_dev );
    
    work[0] = magma_zmake_lwork( lwkopt );
    
    return *info;
} /* magma_zhetrd */


/******************************************************************************/
// TODO info is unused
extern "C" magma_int_t
magma_zhtodhe(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex     *A,   magma_int_t lda,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][10],
    magma_int_t *info)
{
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    magma_int_t k;
    if (uplo == MagmaLower) {
        /* go through each block-column */
        magma_int_t j, jj, jb, mj;
        for (j=0; j < n; j += nb) {
            jj =  j/(nb*ngpu);
            k  = (j/nb)%ngpu;
            
            jb = min(nb, (n-j));
            mj = n-j;
            
            magma_setdevice( k );
            magma_zsetmatrix_async( mj, jb,
                                     A(j,j),         lda,
                                    dA(k, j, jj*nb), ldda,
                                    queues[k][0] );
        }
    }
    else {
        /* go through each block-column */
        magma_int_t j, jj, jb, mj;
        for (j=0; j < n; j += nb) {
            jj =  j/(nb*ngpu);
            k  = (j/nb)%ngpu;
            
            jb = min(nb, (n-j));
            mj = j+jb;
            
            magma_setdevice( k );
            magma_zsetmatrix_async( mj, jb,
                                     A(0, j),        lda,
                                    dA(k, 0, jj*nb), ldda,
                                    queues[k][0] );
        }
    }
    for( k=0; k < ngpu; k++ ) {
        magma_setdevice( k );
        magma_queue_sync( queues[k][0] );
    }
    magma_setdevice( orig_dev );
    
    return *info;
}


/******************************************************************************/
extern "C" void
magma_zher2k_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10])
{
#define dB(id, i, j)  (dB[(id)] + (j)*lddb + (i)+b_offset)
#define dB1(id, i, j) (dB[(id)] + (j)*lddb + (i)+b_offset) + k*lddb
#define dC(id, i, j)  (dC[(id)] + (j)*lddc + (i))

    magma_int_t i, id, ib, ii, kk, n1;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    /* diagonal update */
    for( i=0; i < n; i += nb ) {
        id = ((i+c_offset)/nb)%ngpu;
        kk = (i/(nb*ngpu))%nqueue;
        magma_setdevice( id );

        ib = min(nb, n-i);
        ii = nb*((i+c_offset)/(nb*ngpu));

        /* zher2k on diagonal block */
        trace_gpu_start( id, kk, "syr2k", "syr2k" );
        magma_zher2k( uplo, trans, ib, k,
                      alpha, dB1(id, i,          0 ), lddb,
                             dB(id,  i,          0 ), lddb,
                      beta,  dC(id,  i+c_offset, ii), lddc, queues[id][kk] );
        trace_gpu_end( id, kk );
    }

    /* off-diagonal update */
    if (uplo == MagmaUpper) {
        for( i=nb; i < n; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = (i/(nb*ngpu))%nqueue;
            magma_setdevice( id );
            
            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, i, ib, k,
                         alpha, dB1(id, 0, 0 ), lddb,
                                dB(id,  i, 0 ), lddb,
                         c_one, dC(id,  0, ii), lddc, queues[id][kk] );
        }
    }
    else {
        for( i=0; i < n-nb; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = (i/(nb*ngpu))%nqueue;
            magma_setdevice( id );
            
            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            n1 = n-i-ib;
            
            // zgemm on off-diagonal blocks
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                         alpha, dB1(id, i+ib,          0 ), lddb,
                                dB(id,  i,             0 ), lddb,
                         c_one, dC(id,  i+c_offset+ib, ii), lddc, queues[id][kk] );
            trace_gpu_end( id, kk );
        }
    }

    if (uplo == MagmaUpper) {
        for( i=nb; i < n; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = (i/(nb*ngpu))%nqueue;
            magma_setdevice( id );
            
            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, i, ib, k,
                         alpha, dB( id, 0, 0 ), lddb,
                                dB1(id, i, 0 ), lddb,
                         c_one, dC(id,  0, ii), lddc, queues[id][kk] );
        }
    } else {
        for( i=0; i < n-nb; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = (i/(nb*ngpu))%nqueue;
            magma_setdevice( id );
            
            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            n1 = n-i-ib;
            
            /* zgemm on off-diagonal blocks */
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                         alpha, dB(id,  i+ib,          0 ), lddb,
                                dB1(id, i,             0 ), lddb,
                         c_one, dC(id,  i+c_offset+ib, ii), lddc, queues[id][kk] );
            trace_gpu_end( id, kk );
        }
    }

    for( id=0; id < ngpu; id++ ) {
        magma_setdevice( id );
        for( kk=0; kk < nqueue; kk++ ) {
            magma_queue_sync( queues[id][kk] );
        }
    }
    magma_setdevice( orig_dev );
}
