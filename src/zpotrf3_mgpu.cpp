/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"
#include "trace.h"

#define PRECISION_z

/* === Define what BLAS to use ============================================ */
#if defined(PRECISION_s) || defined(PRECISION_d)
#define ZTRSM_WORK
//#undef  magma_ztrsm
//#define magma_ztrsm magmablas_ztrsm
#endif
/* === End defining what BLAS to use ======================================= */

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.
    Auxiliary subroutine for zpotrf2_ooc. It is multiple gpu interface to compute
    Cholesky of a "rectangular" matrix.

    The factorization has the form
       dA = U**H * U,   if UPLO = MagmaUpper, or
       dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    m       INTEGER
            The number of rows of the submatrix to be factorized.

    @param[in]
    n       INTEGER
            The number of columns of the submatrix to be factorized.

    @param[in]
    off_i   INTEGER
            The first row index of the submatrix to be factorized.

    @param[in]
    off_j   INTEGER
            The first column index of the submatrix to be factorized.

    @param[in]
    nb      INTEGER
            The block size used for the factorization and distribution.

    @param[in,out]
    d_lA    COMPLEX_16 array of pointers on the GPU, dimension (ngpu).
            On entry, the Hermitian matrix dA distributed over GPU.
            (d_lAT[d] points to the local matrix on d-th GPU).
            If UPLO = MagmaLower or MagmaUpper, it respectively uses 
            a 1D block column or row cyclic format (with the block size 
            nb), and each local matrix is stored by column.
            If UPLO = MagmaUpper, the leading N-by-N upper triangular 
            part of dA contains the upper triangular part of the matrix dA, 
            and the strictly lower triangular part of dA is not referenced.  
            If UPLO = MagmaLower, the leading N-by-N lower triangular part 
            of dA contains the lower triangular part of the matrix dA, and 
            the strictly upper triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in,out]
    d_lP    COMPLEX_16 array of pointers on the GPU, dimension (ngpu).
            d_LAT[d] points to workspace of size h*lddp*nb on d-th GPU.

    @param[in]
    lddp    INTEGER
            The leading dimension of the array dP.  LDDA >= max(1,N).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[in,out]
    A       COMPLEX_16 array on the CPU, dimension (LDA,H*NB)
            On exit, the panel is copied back to the CPU

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in]
    h       INTEGER
            It specifies the size of the CPU workspace, A.

    @param[in]
    queues  magma_queue_t
            queues is of dimension (ngpu,3) and contains the queues 
            used for the partial factorization.

    @param[in]
    events  magma_event_t
            events is of dimension(ngpu,5) and contains the events used 
            for the partial factorization.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf3_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDoubleComplex_ptr d_lA[],  magma_int_t ldda,
    magmaDoubleComplex_ptr d_lP[],  magma_int_t lddp,
    magmaDoubleComplex *A,          magma_int_t lda, magma_int_t h,
    magma_queue_t queues[][3], magma_event_t events[][5],
    magma_int_t *info )
{
#define Alo(i, j)  (A +             ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i))
#define Aup(i, j)  (A + (nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i))

#define dlA(id, i, j)     (d_lA[(id)] + (j)*ldda + (i))
#define dlP(id, i, j, k)  (d_lP[(id)] + (k)*nb*lddp + (j)*lddp + (i))
#define dlPT(id, i, j, k) (d_lP[(id)] + (k)*nb*lddp + (j)*nb   + (i))

    magma_int_t     j, jb, nb0, nb2, d, dd, id, j_local, j_local2, buf;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    bool upper = (uplo == MagmaUpper);
    magmaDoubleComplex *dlpanel;
    magma_int_t n_local[MagmaMaxGPUs], ldpanel;
    const magma_int_t stream1 = 0, stream2 = 1, stream3 = 2;
    
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (!upper && ngpu*ldda < max(1,n)) {
        *info = -4;
    } else if (upper && ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
    /* used by ztrsm_work */
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magma_int_t trsm_nb = 128;
    magma_int_t trsm_n = magma_roundup( nb, trsm_nb );
    magmaDoubleComplex *d_dinvA[MagmaMaxGPUs];
    magmaDoubleComplex *dx[MagmaMaxGPUs];
    #define dinvA(d,j) &(d_dinvA[(d)][(j)*trsm_nb*trsm_n])
    #define dx(d,j) &(dx[(d)][(j)*nb*m])
    /*
     * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE
     */
    // TODO free memory on failure.
    magma_int_t dinvA_length = 2*trsm_nb*trsm_n;
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        if ( (MAGMA_SUCCESS != magma_zmalloc( &d_dinvA[d], dinvA_length )) ||
             (MAGMA_SUCCESS != magma_zmalloc( &dx[d],      2*nb*(upper ? n : m) )) ) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
    }
    magma_setdevice(0);
#endif
    
    /* initialization */
    for( d=0; d < ngpu; d++ ) {
        /* local-n and local-ld */
        if (upper) {
            n_local[d] = (n/(nb*ngpu))*nb;
            if (d < (n/nb)%ngpu)
                n_local[d] += nb;
            else if (d == (n/nb)%ngpu)
                n_local[d] += n%nb;
        } else {
            n_local[d] = (m/(nb*ngpu))*nb;
            if (d < (m/nb)%ngpu)
                n_local[d] += nb;
            else if (d == (m/nb)%ngpu)
                n_local[d] += m%nb;
        }
    }

    /* == initialize the trace */
    trace_init( 1, ngpu, 3, (magma_queue_t *)queues );

    if (upper) {
        /* ---------------------------------------------- */
        /* Upper-triangular case                          */
        /* > Compute the Cholesky factorization A = U'*U. */
        /* ---------------------------------------------- */
        for (j=0; j < m; j += nb) {
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%ngpu;
            buf = (j/nb)%ngpu; // right now, we have ngpu buffers, so id and buf are the same..
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*ngpu);
            jb = min(nb, (m-j));
 
            /* Update the current diagonal block on stream1 */
            magma_setdevice(id);
            if ( j > 0 ) {
                trace_gpu_start( id, stream1, "syrk", "syrk" );
                magma_zherk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dlA(id, 0, nb*j_local), ldda,
                            d_one,     dlA(id, j, nb*j_local), ldda,
                            queues[id][stream1]);
                trace_gpu_end( id, stream1 );
            }
            
            /* send the diagonal to cpu on stream1 */
            trace_gpu_start( id, stream1, "comm", "D to CPU" );
            magma_zgetmatrix_async( jb, jb,
                                    dlA(id, j, nb*j_local), ldda,
                                    Aup(j,j),               lda,
                                    queues[id][stream1] );
            trace_gpu_end( id, stream1 );

            /* update off-diagonal blocks in the panel */
            if ( j > 0 ) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2 --;
                    nb0 = nb*j_local2; // number of local columns in the panel, while jb is panel-size (number of rows)
            
                    if ( n_local[d] > nb0 ) {
                        magma_setdevice(d);
                        if ( d == id ) {
                            dlpanel = dlA(d,0,nb*j_local);
                            ldpanel = ldda;
                            // the GPU owns the row from start, and no need of sync.
                            //magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // wait for look-ahead trsm to finish
                        } else {
                            dlpanel = dlP(d,nb,0,buf);
                            ldpanel = lddp;
                            magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                        }
                        trace_gpu_start( d, stream2, "gemm", "gemm" );
                        magma_zgemm(MagmaConjTrans, MagmaNoTrans,
                                    jb, n_local[d]-nb0, j,
                                    c_neg_one, dlpanel,        ldpanel,
                                               dlA(d, 0, nb0), ldda,
                                    c_one,     dlA(d, j, nb0), ldda,
                                    queues[d][stream2]);
                        trace_gpu_end( d, stream2 );
                        magma_event_record( events[d][2], queues[d][stream2] );
                    }
                    d = (d+1)%ngpu;
                }
            }

            /* wait for panel and factorize it on cpu */
            magma_setdevice(id);
            magma_queue_sync( queues[id][stream1] );
            trace_cpu_start( 0, "getrf", "getrf" );
            lapackf77_zpotrf(MagmaUpperStr, &jb, Aup(j,j), &lda, info);
            trace_cpu_end( 0 );
            if (*info != 0) {
                *info = *info + j;
                break;
            }
            
            /* send the diagonal to gpus on stream1 */
            if ( (j+jb) < n) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    if ( d == id ) {
                        dlpanel = dlA(d, j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d,0,0,buf);
                        ldpanel = lddp;
                    }
                    magma_setdevice(d);
                    trace_gpu_start( d, stream1, "comm", "comm" );
                    magma_zsetmatrix_async( jb, jb,
                                            Aup(j,j), lda,
                                            dlpanel,  ldpanel,
                                            queues[d][stream1] );
                    trace_gpu_end( d, stream1 );
                    magma_event_record( events[d][1], queues[d][stream1] );
                    d = (d+1)%ngpu;
                }
            } else {
                magma_setdevice(id);
                trace_gpu_start( id, stream1, "comm", "comm" );
                magma_zsetmatrix_async( jb, jb,
                                        Aup(j,j),               lda,
                                        dlA(id, j, nb*j_local), ldda,
                                        queues[id][stream1] );
                trace_gpu_end( id, stream1 );
            }
            
            /* panel-factorize the off-diagonal */
            if ( (j+jb) < n) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    if ( d == id ) {
                        dlpanel = dlA(d,j,nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d,0,0,buf);
                        ldpanel = lddp;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    
                    magma_setdevice(d);
                    if ( j+jb < m && d == (j/nb+1)%ngpu ) {
                        /* owns the next column, look-ahead next block on stream1 */
                        nb0 = min(nb, nb2);
                        magma_queue_wait_event( queues[d][stream1], events[d][2] ); // wait for gemm update
                        trace_gpu_start( d, stream1, "trsm", "trsm" );
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                            //magmablas_zlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_zlaset( MagmaFull, nb0,     jb,     c_zero, c_zero, dx(d,0), nb0 );
                            magmablas_ztrsm_work( MagmaLeft, MagmaUpper,
                                                  MagmaConjTrans, MagmaNonUnit,
                                                  jb, nb0, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, j, nb*j_local2), ldda,
                                                  dx(d,0), jb,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream1] );
                        #else
                            magma_ztrsm( MagmaLeft, MagmaUpper,
                                         MagmaConjTrans, MagmaNonUnit,
                                         jb, nb0, c_one,
                                         dlpanel,                ldpanel,
                                         dlA(d, j, nb*j_local2), ldda,
                                         queues[d][stream1] );
                        #endif
                        magma_event_record( events[d][4], queues[d][stream1] );
                        trace_gpu_end( d, stream1 );
                    } else if ( nb2 > 0 ) {
                        /* update all the blocks on stream2 */
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for cholesky factor
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                            //magmablas_zlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_zlaset( MagmaFull, nb2,     jb,     c_zero, c_zero, dx(d,0), nb2 );
                            magmablas_ztrsm_work( MagmaLeft, MagmaUpper,
                                                  MagmaConjTrans, MagmaNonUnit,
                                                  jb, nb2, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, j, nb*j_local2), ldda,
                                                  dx(d,0), jb,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream2] );
                        #else
                            magma_ztrsm( MagmaLeft, MagmaUpper,
                                         MagmaConjTrans, MagmaNonUnit,
                                         jb, nb2, c_one,
                                         dlpanel,                ldpanel,
                                         dlA(d, j, nb*j_local2), ldda,
                                         queues[d][stream2] );
                        #endif
                        trace_gpu_end( d, stream2 );
                    }
                    d = (d+1)%ngpu;
                } /* end of for */

                /* ========================================================== */
                if ( j+jb < m ) {
                    d = (j/nb+1)%ngpu;
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    nb0 = min(nb, n_local[d]-nb*j_local2 );
                
                    /* even on 1 gpu, off-diagonals are copied to cpu (synchronize at the end).      *
                     * so we have the Cholesky factor, but only diagonal submatrix of the big panel, *
                     * on cpu at the end.                                                            */
                    magma_int_t d2, buf2;
                    magma_setdevice(d);
                    /* lookahead done */
                    magma_queue_wait_event( queues[d][stream3], events[d][4] );
                
                    trace_gpu_start( d, stream3, "comm", "row to CPU" );
                    magma_zgetmatrix_async( (j+jb), nb0,
                                            dlA(d, 0, nb*j_local2), ldda,
                                            Aup(0,j+jb),            lda,
                                            queues[d][stream3] );
                    trace_gpu_end( d, stream3 );
                    magma_event_record( events[d][3], queues[d][stream3] );
                    /* needed on pluto */
                    //magma_queue_sync( queues[d][stream3] );
                
                    /* broadcast rows to gpus on stream2 */
                    buf2 = ((j+jb)/nb)%ngpu;
                    for( d2=0; d2 < ngpu; d2++ ) {
                        if ( d2 != d ) {
                            magma_setdevice(d2);
                            trace_gpu_start( d2, stream3, "comm", "row to GPUs" );
                            magma_queue_wait_event( queues[d2][stream3], events[d][3] ); // rows arrived at cpu on stream3
                            magma_zsetmatrix_async( j+jb, nb0,
                                                    Aup(0,j+jb),       lda,
                                                    dlP(d2,nb,0,buf2), lddp,
                                                    queues[d2][stream3] );
                            trace_gpu_end( d2, stream3 );
                            magma_event_record( events[d2][0], queues[d2][stream3] );
                        }
                    }

                    /* =========================== */
                    /* update the remaining blocks */
                    nb2 = n_local[d]-(nb*j_local2 + nb0);
                    if ( nb2 > 0 ) {
                        if ( d == id ) {
                            dlpanel = dlA(d, j, nb*j_local);
                            ldpanel = ldda;
                        } else {
                            dlpanel = dlP(d,0,0,buf);
                            ldpanel = lddp;
                        }
                        magma_setdevice(d);
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                            bool flag = 0;
                            if (flag == 0) {
                                magma_queue_wait_event( queues[d][stream2], events[d][4] ); // lookahead -> diagonal inversion
                            } else {
                                magmablas_zlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,flag), trsm_nb, queues[d][stream2] );
                                magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                            }
                            magmablas_zlaset( MagmaFull, nb2, jb, c_zero, c_zero, dx(d,1), nb2, queues[d][stream2] );
                            magmablas_ztrsm_work( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                                  jb, nb2, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, j, nb*j_local2+nb0), ldda,
                                                  dx(d,1), jb,
                                                  flag, dinvA(d,flag), dinvA_length,
                                                  queues[d][stream2] );
                        #else
                            magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for cholesky factor
                            magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                         jb, nb2, c_one,
                                         dlpanel, ldpanel,
                                         dlA(d, j, nb*j_local2+nb0), ldda,
                                         queues[d][stream2] );
                        #endif
                        trace_gpu_end( d, stream2 );
                    }
                }
            } /* end of ztrsm */
        } /* end of for j=1, .., n */
    } else {
        /* ---------------------------------------------- */
        /* Lower-triangular case                          */
        /* > Compute the Cholesky factorization A = L*L'. */
        /* ---------------------------------------------- */
        for (j=0; j < n; j += nb) {
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%ngpu;
            buf = (j/nb)%ngpu;
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*ngpu);
            jb = min(nb, (n-j));

            /* Update the current diagonal block on stream1 */
            magma_setdevice(id);
            if ( j > 0 ) {
                magma_zherk( MagmaLower, MagmaNoTrans, jb, j,
                             d_neg_one, dlA(id, nb*j_local, 0), ldda,
                             d_one,     dlA(id, nb*j_local, j), ldda,
                             queues[id][stream1] );
            }

            /* send the diagonal to cpu on stream1 */
            magma_zgetmatrix_async( jb, jb,
                                    dlA(id, nb*j_local, j), ldda,
                                    Alo(j,j),               lda,
                                    queues[id][stream1] );

            /* update off-diagonal blocks of the panel */
            if ( j > 0 ) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
            
                    if ( nb0 < n_local[d] ) {
                        magma_setdevice(d);
                        if ( d == id ) {
                            dlpanel = dlA(d, nb*j_local, 0);
                            ldpanel = ldda;
                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // wait for look-ahead trsm to finish
                        } else {
                            dlpanel = dlPT(d,0,nb,buf);
                            ldpanel = nb;
                            magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                        }
                        magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                     n_local[d]-nb0, jb, j,
                                     c_neg_one, dlA(d, nb0, 0), ldda,
                                                dlpanel,        ldpanel,
                                     c_one,     dlA(d, nb0, j), ldda,
                                     queues[d][stream2] );
                        magma_event_record( events[d][2], queues[d][stream2] );
                    }
                    d = (d+1)%ngpu;
                }
            }

            /* wait for the panel and factorized it on cpu */
            magma_setdevice(id);
            magma_queue_sync( queues[id][stream1] );
            lapackf77_zpotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus on stream1 */
            if ( (j+jb) < m) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    if ( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    magma_setdevice(d);
                    magma_zsetmatrix_async( jb, jb,
                                            Alo(j,j), lda,
                                            dlpanel,  ldpanel,
                                            queues[d][stream1] );
                    magma_event_record( events[d][1], queues[d][stream1] );
                    d = (d+1)%ngpu;
                }
            } else {
                magma_setdevice(id);
                magma_zsetmatrix_async( jb, jb,
                                        Alo(j,j),               lda,
                                        dlA(id, nb*j_local, j), ldda,
                                        queues[id][stream1] );
            }

            /* panel factorize the off-diagonal */
            if ( (j+jb) < m) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    if ( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2);
                    
                    magma_setdevice(d);
                    if ( j+nb < n && d == (j/nb+1)%ngpu ) { /* owns next column, look-ahead next block on stream1 */
                        if ( j > 0 ) magma_queue_wait_event( queues[d][stream1], events[d][2] ); // wait for gemm update
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                            //magmablas_zlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_zlaset( MagmaFull, nb0,     jb,     c_zero, c_zero, dx(d,0), nb0 );
                            magmablas_ztrsm_work( MagmaRight, MagmaLower,
                                                  MagmaConjTrans, MagmaNonUnit,
                                                  nb0, jb, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, nb*j_local2, j), ldda,
                                                  dx(d,0), nb0,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream1] );
                        #else
                            magma_ztrsm( MagmaRight, MagmaLower,
                                         MagmaConjTrans, MagmaNonUnit,
                                         nb0, jb, c_one,
                                         dlpanel, ldpanel,
                                         dlA(d, nb*j_local2, j), ldda,
                                         queues[d][stream1] );
                        #endif
                        magma_event_record( events[d][4], queues[d][stream1] );
                    } else if ( nb2 > 0 ) { /* other gpus updating all the blocks on stream2 */
                        /* update the entire column */
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for the cholesky factor
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                            //magmablas_zlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_zlaset( MagmaFull, nb2,     jb,     c_zero, c_zero, dx(d,0), nb2 );
                            magmablas_ztrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                                  nb2, jb, c_one,
                                                  dlpanel,                ldpanel,
                                                  dlA(d, nb*j_local2, j), ldda,
                                                  dx(d,0), nb2,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream2] );
                        #else
                            magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                         nb2, jb, c_one,
                                         dlpanel,                ldpanel,
                                         dlA(d, nb*j_local2, j), ldda,
                                         queues[d][stream2] );
                        #endif
                    }
                    d = (d+1)%ngpu;
                } /* end for d */

                /* ========================================================== */
                if ( j+jb < n ) {
                    d = (j/nb+1)%ngpu;
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    nb0 = min(nb, n_local[d]-nb*j_local2 );
                
                    /* even on 1 gpu, we copy off-diagonal to cpu (but don't synchronize).  */
                    /* so we have the Cholesky factor on cpu at the end.                    */
                    magma_int_t d2, buf2;
//#define ZPOTRF_DEVICE_TO_DEVICE
#ifdef ZPOTRF_DEVICE_TO_DEVICE
                    // lookahead done
                
                    /* broadcast the rows to gpus */
                    buf2 = ((j+jb)/nb)%ngpu;
                    for( d2=0; d2 < ngpu; d2++ ) {
                        magma_setdevice(d2);
                        magma_queue_wait_event( queues[d2][stream3], events[d][4] );
                        if ( d2 != d ) {
                            magma_zcopymatrix_async( nb0, j+jb,
                                                     dlPT(d2,0,nb,buf2), nb, // first nbxnb reserved for diagonal block
                                                     dlA(d, nb*j_local2, 0), ldda,
                                                     queues[d2][stream3] );
                            magma_event_record( events[d2][0], queues[d2][stream3] );
                        } else {
                            magma_zgetmatrix_async( nb0, j+jb,
                                                    dlA(d, nb*j_local2, 0), ldda,
                                                    Alo(j+jb,0),            lda,
                                                    queues[d][stream3] );
                        }
                    }
#else
                    // lookahead done
                    magma_setdevice(d);
                    magma_queue_wait_event( queues[d][stream3], events[d][4] );
                    magma_zgetmatrix_async( nb0, j+jb,
                                            dlA(d, nb*j_local2, 0), ldda,
                                            Alo(j+jb,0),            lda,
                                            queues[d][stream3] );
                    magma_event_record( events[d][3], queues[d][stream3] );
                    /* syn on rows on CPU, seem to be needed on Pluto */
                    //magma_queue_sync( queues[d][stream3] );
                
                    /* broadcast the rows to gpus */
                    buf2 = ((j+jb)/nb)%ngpu;
                    for( d2=0; d2 < ngpu; d2++ ) {
                        if ( d2 != d ) {
                            magma_setdevice(d2);
                            magma_queue_wait_event( queues[d2][stream3], events[d][3] ); // getmatrix done
                            magma_zsetmatrix_async( nb0, j+jb,
                                                    Alo(j+jb,0),        lda,
                                                    dlPT(d2,0,nb,buf2), nb, // first nbxnb reserved for diagonal block
                                                    queues[d2][stream3] );
                            magma_event_record( events[d2][0], queues[d2][stream3] );
                        }
                    }
#endif
                    /* =================================== */
                    /* updates remaining blocks on stream2 */
                    nb2 = n_local[d] - (j_local2*nb + nb0);
                    if ( nb2 > 0 ) {
                        if ( d == id ) {
                            dlpanel = dlA(d, nb*j_local, j);
                            ldpanel = ldda;
                        } else {
                            dlpanel = dlPT(d,0,0,buf);
                            ldpanel = nb;
                        }
                        magma_setdevice(d);
                        /* update the remaining blocks in the column */
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                            bool flag = 0;
                            if (flag == 0) {
                                magma_queue_wait_event( queues[d][stream2], events[d][4] ); // lookahead -> diagonal inversion
                            } else {
                                magmablas_zlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,flag), trsm_nb, queues[d][stream2] );
                                magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                            }
                            magmablas_zlaset( MagmaFull, nb2, jb, c_zero, c_zero, dx(d,1), nb2, queues[d][stream2] );
                            magmablas_ztrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                                  nb2, jb, c_one,
                                                  dlpanel,                    ldpanel,
                                                  dlA(d, nb*j_local2+nb0, j), ldda,
                                                  dx(d,1), nb2,
                                                  flag, dinvA(d,flag), dinvA_length,
                                                  queues[d][stream2] );
                        #else
                            magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                            magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                         nb2, jb, c_one,
                                         dlpanel,                    ldpanel,
                                         dlA(d, nb*j_local2+nb0, j), ldda,
                                         queues[d][stream2] );
                        #endif
                    }
                }
            }
        }
    } /* end of else not upper */

    /* == finalize the trace == */
    trace_finalize( "zpotrf.svg", "trace.css" );
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        for( j=0; j < 3; j++ ) {
            magma_queue_sync( queues[d][j] );
        }
        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
        magma_free( d_dinvA[d] );
        magma_free( dx[d] );
        #endif
    }
    magma_setdevice( orig_dev );

    return *info;
} /* magma_zpotrf_mgpu */

#undef Alo
#undef Aup
#undef dlA
#undef dlP
#undef dlPT


#define A(i, j)  (A +(j)*lda  + (i))
#define dA(d, i, j) (dA[(d)]+(j)*ldda + (i))


/******************************************************************************/
extern "C" magma_int_t
magma_zhtodpo(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDoubleComplex    *A,    magma_int_t lda,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info)
{
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    magma_int_t k;
    if (uplo == MagmaUpper) {
        magma_int_t j, jj, jb, mj;
        
        /* go through each column */
        for (j=off_j; j < n; j += nb) {
            jj = (j-off_j)/(nb*ngpu);
            k  = ((j-off_j)/nb)%ngpu;
            
            jb = min(nb, (n-j));
            if (j+jb < off_j+m)
                mj = (j-off_i)+jb;
            else
                mj = m;

            magma_setdevice(k);
            magma_zsetmatrix_async( mj, jb,
                                    A(off_i, j),     lda,
                                    dA(k, 0, jj*nb), ldda,
                                    queues[k][0] );
        }
    }
    else {
        magma_int_t i, ii, ib, ni;
        
        /* go through each row */
        for (i=off_i; i < m; i += nb) {
            ii = (i-off_i)/(nb*ngpu);
            k  = ((i-off_i)/nb)%ngpu;
            
            ib = min(nb, (m-i));
            if (i+ib < off_i+n)
                ni = (i-off_i)+ib;
            else
                ni = n;
            
            magma_setdevice(k);
            magma_zsetmatrix_async( ib, ni,
                                    A(i, off_j),     lda,
                                    dA(k, ii*nb, 0), ldda,
                                    queues[k][0] );
        }
    }
    for( k=0; k < ngpu; k++ ) {
        magma_setdevice(k);
        magma_queue_sync( queues[k][0] );
    }
    magma_setdevice( orig_dev );

    return *info;
}


/******************************************************************************/
extern "C" magma_int_t
magma_zdtohpo(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    magmaDoubleComplex    *A,    magma_int_t lda,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info)
{
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    magma_int_t k;
    if (uplo == MagmaUpper) {
        magma_int_t j, jj, jb, mj;
        
        /* go through each column */
        for (j=off_j+NB; j < n; j += nb) {
            jj =  (j-off_j)/(nb*ngpu);
            k  = ((j-off_j)/nb)%ngpu;
            
            jb = min(nb, (n-j));
            if (j+jb < off_j+m)
                mj = (j-off_i)+jb;
            else
                mj = m;

            magma_setdevice(k);
            magma_zgetmatrix_async( mj, jb,
                                    dA(k, 0, jj*nb), ldda,
                                    A(off_i, j),     lda,
                                    queues[k][0] );
            magma_queue_sync( queues[k][0] );
        }
    } else {
        magma_int_t i, ii, ib, ni;
        
        /* go through each row */
        for (i=off_i+NB; i < m; i += nb) {
            ii = (i-off_i)/(nb*ngpu);
            k  = ((i-off_i)/nb)%ngpu;
            
            ib = min(nb, (m-i));
            if (i+ib < off_i+n)
                ni = (i-off_i)+ib;
            else
                ni = n;
            
            magma_setdevice(k);
            magma_zgetmatrix_async( ib, ni,
                                    dA(k, ii*nb, 0), ldda,
                                    A(i, off_j),     lda,
                                    queues[k][0] );
            magma_queue_sync( queues[k][0] );
        }
    }
    /*for( k=0; k < ngpu; k++ ) {
        magma_setdevice(k);
        magma_queue_sync( queues[k][0] );
    }*/
    magma_setdevice( orig_dev );

    return *info;
}

#undef A
#undef dA
