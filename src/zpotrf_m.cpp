/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

//#include "../testing/flops.h"
#include "magma_timer.h"

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine. The matrix A may exceed the GPU memory.

    The factorization has the form
       A = U**H * U,   if UPLO = MagmaUpper, or
       A = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    uplo     magma_uplo_t
      -      = MagmaUpper:  Upper triangle of A is stored;
      -      = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n        INTEGER
             The order of the matrix A.  N >= 0.

    @param[in,out]
    A        COMPLEX_16 array, dimension (LDA,N)
             On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
             N-by-N upper triangular part of A contains the upper
             triangular part of the matrix A, and the strictly lower
             triangular part of A is not referenced.  If UPLO = MagmaLower, the
             leading N-by-N lower triangular part of A contains the lower
             triangular part of the matrix A, and the strictly upper
             triangular part of A is not referenced.
    \n
             On exit, if INFO = 0, the factor U or L from the Cholesky
             factorization A = U**H * U or A = L * L**H.
    \n
             Higher performance is achieved if A is in pinned memory, e.g.
             allocated using magma_malloc_pinned.

    @param[in]
    lda      INTEGER
             The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    info     INTEGER
      -      = 0:  successful exit
      -      < 0:  if INFO = -i, the i-th argument had an illegal value
                   or another error occured, such as memory allocation failed.
      -      > 0:  if INFO = i, the leading minor of order i is not
                   positive definite, and the factorization could not be
                   completed.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_m(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info)
{
#define    A(i, j)    (    A      + (j)*lda   + (i))
#define   dA(d, i, j) (dwork[(d)] + (j)*lddla + (i))
#define   dT(d, i, j) (   dt[(d)] + (j)*ldda  + (i))
#define dAup(d, i, j) (dwork[(d)] + (j)*NB    + (i))
#define dTup(d, i, j) (   dt[(d)] + (j)*nb    + (i))

    /* Local variables */
    double                 d_one     =  1.0;
    double                 d_neg_one = -1.0;
    magmaDoubleComplex     c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex     c_neg_one = MAGMA_Z_NEG_ONE;
    const char* uplo_  = lapack_uplo_const( uplo  );
    bool upper = (uplo == MagmaUpper);

    magmaDoubleComplex *dwork[MagmaMaxGPUs], *dt[MagmaMaxGPUs];
    magma_int_t     ldda, lddla, nb, iinfo, n_local[MagmaMaxGPUs], J2, d, ngpu0 = ngpu;
    magma_int_t     j, jj, jb, J, JB, NB, h;
    magma_queue_t   queues[MagmaMaxGPUs][3];
    magma_event_t   event[MagmaMaxGPUs][5];
    magma_timer_t time_total=0, time_sum=0, time=0;
    
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    nb = magma_get_dpotrf_nb(n);
    if ( ngpu0 > n/nb ) {
        ngpu = n/nb;
        if ( n%nb != 0 ) ngpu ++;
    } else {
        ngpu = ngpu0;
    }
    //ldda  = magma_roundup( n, 32 );
    ldda  = magma_roundup( n, nb );
    lddla = magma_roundup( nb*((n+nb*ngpu-1)/(nb*ngpu)), 32 );

    /* figure out NB */
    size_t freeMem, totalMem;
    magma_mem_info(&freeMem, &totalMem);
    freeMem /= sizeof(magmaDoubleComplex);
    
    //MB = n;  /* number of rows in the big panel    */
    NB = (magma_int_t)((0.8*freeMem - max(2,ngpu)*nb*ldda - (n+nb)*nb)/lddla); /* number of columns in the big panel */
    //NB = min(5*nb,n);

    if ( NB >= n ) {
        #ifdef CHECK_ZPOTRF_OOC
        printf( "      * still fits in GPU memory.\n" );
        #endif
        NB = n;
    } else {
        #ifdef CHECK_ZPOTRF_OOC
        printf( "      * doesn't fit in GPU memory.\n" );
        #endif
        NB = (NB/nb) * nb;   /* making sure it's devisable by nb   */
    }
    #ifdef CHECK_ZPOTRF_OOC
    if ( NB != n ) printf( "      * running in out-core mode (n=%lld, NB=%lld, nb=%lld, lddla=%lld, freeMem=%.2e).\n", (long long) n, (long long) NB, (long long) nb, (long long) lddla, (double) freeMem );
    else           printf( "      * running in in-core mode  (n=%lld, NB=%lld, nb=%lld, lddla=%lld, freeMem=%.2e).\n", (long long) n, (long long) NB, (long long) nb, (long long) lddla, (double) freeMem );
    fflush(stdout);
    #endif
    for (d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        if (MAGMA_SUCCESS != magma_zmalloc( &dt[d], NB*lddla + max(2,ngpu)*nb*ldda )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        dwork[d] = &dt[d][max(2,ngpu)*nb*ldda];
        
        for( j=0; j < 3; j++ )
            magma_queue_create( d, &queues[d][j] );
        for( j=0; j < 5; j++ )
            magma_event_create( &event[d][j]  );
    }
    magma_setdevice(0);

    timer_start( time_total );

    if (nb <= 1 || nb >= n) {
        lapackf77_zpotrf(uplo_, &n, A, &lda, info);
    } else {

    /* Use hybrid blocked code. */
    if (upper) {
        /* =========================================================== *
         * Compute the Cholesky factorization A = U'*U.                *
         * big panel is divided by block-row and distributed in block  *
         * column cyclic format                                        */
        
        /* for each big-panel */
        for( J=0; J < n; J += NB ) {
            JB = min(NB,n-J);
            if ( ngpu0 > (n-J)/nb ) {
                ngpu = (n-J)/nb;
                if ( (n-J)%nb != 0 ) ngpu ++;
            } else {
                ngpu = ngpu0;
            }
            
            /* load the new big-panel by block-rows */
            magma_zhtodpo( ngpu, uplo, JB, n, J, J, nb, A, lda, dwork, NB, queues, &iinfo);
            
            /* update with the previous big-panels */
            timer_start( time );
            for( j=0; j < J; j += nb ) {
                /* upload the diagonal of the block column (broadcast to all GPUs) */
                for( d=0; d < ngpu; d++ ) {
                    magma_setdevice(d);
                    magma_zsetmatrix_async( nb, JB,
                                            A(j, J),       lda,
                                            dTup(d, 0, J), nb,
                                            queues[d][0] );
                    n_local[d] = 0;
                }
                
                /* distribute off-diagonal blocks to GPUs */
                for( jj=J+JB; jj < n; jj += nb ) {
                    d  = ((jj-J)/nb)%ngpu;
                    magma_setdevice(d);
                    
                    jb = min(nb, n-jj);
                    magma_zsetmatrix_async( nb, jb,
                                            A(j, jj),                    lda,
                                            dTup(d, 0, J+JB+n_local[d]), nb,
                                            queues[d][0] );
                    n_local[d] += jb;
                }
                
                /* wait for the communication */
                for( d=0; d < ngpu; d++ ) {
                    magma_setdevice(d);
                    magma_queue_sync( queues[d][0] );
                }
                
                /* update the current big-panel using the previous block-row */
                /* -- process the big diagonal block of the big panel */
                for( jj=0; jj < JB; jj += nb ) { // jj is 'local' column index within the big panel
                    d  = (jj/nb)%ngpu;
                    J2 = jj/(nb*ngpu);
                    
                    magma_setdevice(d);
                    J2 = nb*J2;

                    jb = min(nb,JB-jj); // number of columns in this current block-row
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                                 jj, jb, nb,
                                 c_neg_one, dTup(d, 0, J   ), nb,
                                            dTup(d, 0, J+jj), nb,
                                 c_one,     dAup(d, 0, J2), NB,
                                 queues[d][J2%2] );
                    
                    magma_zherk( MagmaUpper, MagmaConjTrans, jb, nb,
                                 d_neg_one, dTup(d, 0,  J+jj), nb,
                                 d_one,     dAup(d, jj, J2), NB,
                                 queues[d][J2%2] );
                }
                /* -- process the remaining big off-diagonal block of the big panel */
                if ( n > J+JB ) {
                    for( d=0; d < ngpu; d++ ) {
                        magma_setdevice(d);
                        
                        /* local number of columns in the big panel */
                        n_local[d] = ((n-J)/(nb*ngpu))*nb;
                        if (d < ((n-J)/nb)%ngpu)
                            n_local[d] += nb;
                        else if (d == ((n-J)/nb)%ngpu)
                            n_local[d] += (n-J)%nb;
                        
                        /* subtracting the local number of columns in the diagonal */
                        J2 = nb*(JB/(nb*ngpu));
                        if ( d < (JB/nb)%ngpu )
                            J2 += nb;

                        n_local[d] -= J2;
                        
                        magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                                     JB, n_local[d], nb,
                                     c_neg_one, dTup(d, 0, J   ), nb,
                                                dTup(d, 0, J+JB), nb,
                                     c_one,     dAup(d, 0, J2), NB,
                                     queues[d][2] );
                    }
                }
                
                /* wait for the previous updates */
                for( d=0; d < ngpu; d++ ) {
                    magma_setdevice(d);
                    for( jj=0; jj < 3; jj++ )
                        magma_queue_sync( queues[d][jj] );
                }
                magma_setdevice(0);
            } /* end of updates with previous rows */
            
            /* factor the big panel */
            h  = magma_ceildiv( JB, nb ); // big diagonal of big panel will be on CPU
            // using three queues
            magma_zpotrf3_mgpu(ngpu, uplo, JB, n-J, J, J, nb,
                               dwork, NB, dt, ldda, A, lda, h, queues, event, &iinfo);
            if ( iinfo != 0 ) {
                *info = J+iinfo;
                break;
            }
            time_sum += timer_stop( time );
            
            /* upload the off-diagonal (and diagonal!!!) big panel */
            magma_zdtohpo(ngpu, uplo, JB, n, J, J, nb, NB, A, lda, dwork, NB, queues, &iinfo);
            //magma_zdtohpo(ngpu, uplo, JB, n, J, J, nb, 0, A, lda, dwork, NB, queues, &iinfo);
        }
    } else {
        /* ========================================================= *
         * Compute the Cholesky factorization A = L*L'.              */
        
        /* for each big-panel */
        for( J=0; J < n; J += NB ) {
            JB = min(NB,n-J);
            if ( ngpu0 > (n-J)/nb ) {
                ngpu = (n-J)/nb;
                if ( (n-J)%nb != 0 ) ngpu ++;
            } else {
                ngpu = ngpu0;
            }
            
            /* load the new big-panel by block-columns */
            magma_zhtodpo( ngpu, uplo, n, JB, J, J, nb, A, lda, dwork, lddla, queues, &iinfo);
            
            /* update with the previous big-panels */
            timer_start( time );
            for( j=0; j < J; j += nb ) {
                /* upload the diagonal of big panel */
                for( d=0; d < ngpu; d++ ) {
                    magma_setdevice(d);
                    magma_zsetmatrix_async( JB, nb,
                                            A(J, j),     lda,
                                            dT(d, J, 0), ldda,
                                            queues[d][0] );
                    n_local[d] = 0;
                }
                
                /* upload off-diagonals */
                for( jj=J+JB; jj < n; jj += nb ) {
                    d  = ((jj-J)/nb)%ngpu;
                    magma_setdevice(d);
                    
                    jb = min(nb, n-jj);
                    magma_zsetmatrix_async( jb, nb,
                                            A(jj, j),                  lda,
                                            dT(d, J+JB+n_local[d], 0), ldda,
                                            queues[d][0] );
                    n_local[d] += jb;
                }
                
                /* wait for the communication */
                for( d=0; d < ngpu; d++ ) {
                    magma_setdevice(d);
                    magma_queue_sync( queues[d][0] );
                }
                
                /* update the current big-panel using the previous block-row */
                for( jj=0; jj < JB; jj += nb ) { /* diagonal */
                    d  = (jj/nb)%ngpu;
                    J2 = jj/(nb*ngpu);
                    
                    magma_setdevice(d);
                    
                    J2 = nb*J2;
                    jb = min(nb,JB-jj);
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 jb, jj, nb,
                                 c_neg_one, dT(d, J+jj, 0), ldda,
                                            dT(d, J,    0), ldda,
                                 c_one,     dA(d, J2,   0), lddla,
                                 queues[d][J2%2] );
                    
                    magma_zherk( MagmaLower, MagmaNoTrans, jb, nb,
                                 d_neg_one, dT(d, J+jj, 0), ldda,
                                 d_one,     dA(d, J2,  jj), lddla,
                                 queues[d][J2%2] );
                }
                
                if ( n > J+JB ) { /* off-diagonal */
                    for( d=0; d < ngpu; d++ ) {
                        magma_setdevice(d);
                        
                        /* local number of columns in the big panel */
                        n_local[d] = (((n-J)/nb)/ngpu)*nb;
                        if (d < ((n-J)/nb)%ngpu)
                            n_local[d] += nb;
                        else if (d == ((n-J)/nb)%ngpu)
                            n_local[d] += (n-J)%nb;
                        
                        /* subtracting local number of columns in diagonal */
                        J2 = nb*(JB/(nb*ngpu));
                        if ( d < (JB/nb)%ngpu )
                            J2 += nb;

                        n_local[d] -= J2;
                        
                        magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                     n_local[d], JB, nb,
                                     c_neg_one, dT(d, J+JB, 0), ldda,
                                                dT(d, J,    0), ldda,
                                     c_one,     dA(d, J2,   0), lddla,
                                     queues[d][2] );
                    }
                }
                /* wait for the previous updates */
                for( d=0; d < ngpu; d++ ) {
                    magma_setdevice(d);
                    for( jj=0; jj < 3; jj++ )
                        magma_queue_sync( queues[d][jj] );
                }
                magma_setdevice(0);
            }
            
            /* factor the big panel */
            h = magma_ceildiv( JB, nb ); // big diagonal of big panel will be on CPU
            // using three queues
            magma_zpotrf3_mgpu(ngpu, uplo, n-J, JB, J, J, nb,
                               dwork, lddla, dt, ldda, A, lda, h, queues, event, &iinfo);
            if ( iinfo != 0 ) {
                *info = J+iinfo;
                break;
            }
            time_sum += timer_stop( time );
            
            /* upload the off-diagonal big panel */
            magma_zdtohpo( ngpu, uplo, n, JB, J, J, nb, JB, A, lda, dwork, lddla, queues, &iinfo);
        } /* end of for J */
    } /* if upper */
    } /* if nb */
    timer_stop( time_total );
    
    if ( ngpu0 > n/nb ) {
        ngpu = n/nb;
        if ( n%nb != 0 ) ngpu ++;
    } else {
        ngpu = ngpu0;
    }
    for (d=0; d < ngpu; d++ ) {
        magma_setdevice(d);

        for( j=0; j < 3; j++ ) {
            magma_queue_destroy( queues[d][j] );
        }
        magma_free( dt[d] );

        for( j=0; j < 5; j++ ) {
            magma_event_destroy( event[d][j] );
        }
    }
    magma_setdevice( orig_dev );
    
    // timer_printf( "\n n=%lld NB=%lld nb=%lld\n", (long long) n, (long long) NB, (long long) nb );
    // timer_printf( " Without memory allocation: %f / %f = %f GFlop/s\n",
    //               FLOPS_ZPOTRF(n) / 1e9,  time_total,
    //               FLOPS_ZPOTRF(n) / 1e9 / time_total );
    // timer_printf( " Performance %f / %f = %f GFlop/s\n",
    //               FLOPS_ZPOTRF(n) / 1e9,  time_sum,
    //               FLOPS_ZPOTRF(n) / 1e9 / time_sum );
    
    return *info;
} /* magma_zpotrf_ooc */

#undef A
#undef dA
#undef dT
#undef dAup
#undef dTup
