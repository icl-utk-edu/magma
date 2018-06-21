/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "trace.h"


/***************************************************************************//**
    Purpose
    -------
    ZHETRD_HE2HB reduces a complex Hermitian matrix A to real symmetric
    band-diagonal form T by an orthogonal similarity transformation:
    Q**H * A * Q = T.
    This version stores the triangular matrices T used in the accumulated
    Householder transformations (I - V T V').

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    nb      INTEGER
            The inner blocking.  nb >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = MagmaUpper, the Upper band-diagonal of A is
            overwritten by the corresponding elements of the
            band-diagonal matrix T, and the elements above the band
            diagonal, with the array TAU, represent the orthogonal
            matrix Q as a product of elementary reflectors; if UPLO
            = MagmaLower, the the Lower band-diagonal of A is overwritten by
            the corresponding elements of the band-diagonal
            matrix T, and the elements below the band-diagonal, with
            the array TAU, represent the orthogonal matrix Q as a product
            of elementary reflectors. See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    tau     COMPLEX_16 array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= 1.
            For optimum performance LWORK >= N*NB, where NB is the
            optimal blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    dT      COMPLEX_16 array on the GPU, dimension N*NB,
            where NB is the optimal blocksize.
            On exit dT holds the upper triangular matrices T from the
            accumulated Householder transformations (I - V T V') used
            in the factorization. The nb x nb matrices T are ordered
            consecutively in memory one after another.

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

    @ingroup magma_hetrd_he2hb
*******************************************************************************/
extern "C" magma_int_t
magma_zhetrd_he2hb(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info)
{
    #ifdef HAVE_clBLAS
    #define dA(a_1,a_2)  (dA, (dA_offset + ((a_2)-1)*(ldda) + (a_1)-1))
    #define dT(a_1)      (dT, (dT_offset + ((a_1)-1)*(lddt)))
    #else
    #define dA(a_1,a_2)  (dA + ((a_2)-1)*(ldda) + (a_1)-1)
    #define dT(a_1)      (dT + ((a_1)-1)*(lddt))
    #endif

    #define  A(a_1,a_2)  ( A + ((a_2)-1)*( lda) + (a_1)-1)
    #define tau_ref(a_1) (tau + (a_1)-1)

    magma_int_t ldda = magma_roundup( n, 32 );
    magma_int_t lddt = nb;
   
    magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_neg_half = MAGMA_Z_NEG_HALF;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    double  d_one = MAGMA_D_ONE;

    magma_int_t pm, pn, indi, indj, pk;
    magma_int_t pm_old=0, pn_old=0, indi_old=0, indj_old=0;

    magma_int_t i;
    magma_int_t lwkopt;

    *info = 0;
    bool upper = (uplo == MagmaUpper);
    bool lquery = (lwork == -1);
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    } else if (lwork < 1 && ! lquery) {
        *info = -9;
    }

    /* Determine the block size. */
    lwkopt = n * nb;
    if (*info == 0) {
        work[0] = magma_zmake_lwork( lwkopt );
    }

    if (*info != 0)
        return *info;
    else if (lquery)
        return *info;

    /* Quick return if possible */
    if (n == 0) {
        work[0] = c_one;
        return *info;
    }

    magmaDoubleComplex *dA;
    if (MAGMA_SUCCESS != magma_zmalloc( &dA, (n + 2*nb)*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    // limit to 16 threads
    magma_int_t orig_threads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads( min(orig_threads,16) );

    /* Use the first panel of dA as work space */
    magmaDoubleComplex *dwork = dA + n*ldda;
    magmaDoubleComplex *dW    = dwork + nb*ldda;

    #ifdef TRACING
    char buf[80];
    #endif
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    
    trace_init( 1, 1, 3, queues );

    lwork -= nb*nb;
    magmaDoubleComplex *hT = work + lwork;
    memset( hT, 0, nb*nb*sizeof(magmaDoubleComplex));

    magma_event_t Pupdate_event;
    cudaEventCreateWithFlags(&Pupdate_event,cudaEventDisableTiming);
    //magma_event_create(&Pupdate_event);


    if (upper) {
        printf("ZHETRD_HE2HB is not yet implemented for upper matrix storage. Exit.\n");
        return MAGMA_ERR_NOT_IMPLEMENTED;
    } else {
        /* Copy the matrix to the GPU */
        if (1 <= n-nb) {
            trace_gpu_start( 0, 0, "set", "set A" );
            magma_zsetmatrix_async( (n-nb), (n-nb),
                                    A(nb+1, nb+1),  lda,
                                    dA(nb+1, nb+1), ldda, queues[0] );
            trace_gpu_end( 0, 0 );
        }

        /* Reduce the lower triangle of A */
        for (i = 1; i <= n-nb; i += nb) {
            indi = i+nb;
            indj = i;
            pm   = n - i - nb + 1;
            //pn   = min(i+nb-1, n-nb) -i + 1;
            pn   = nb;
            
            /*   Get the current panel (no need for the 1st iteration) */
            if (i > 1 ) {
                // magma_zpanel_to_q copy the upper oof diagonal part of
                // the matrix to work to be restored later. acctually
                //  the zero's and one's putted are not used this is only
                //   because we don't have a function that copy only the
                //    upper part of A to be restored after copying the
                //    lookahead panel that has been computted from GPU to CPU.
                magma_zpanel_to_q(MagmaUpper, pn-1, A(i, i+1), lda, work);

                trace_gpu_start( 0, 1, "get", "get panel" );
                //magma_queue_sync( queues[0] );
                magma_queue_wait_event(queues[1], Pupdate_event);  //, 0);
                magma_zgetmatrix_async( (pm+pn), pn,
                                        dA( i, i), ldda,
                                        A ( i, i), lda, queues[1] );
                trace_gpu_end( 0, 1 );

                trace_gpu_start( 0, 2, "her2k", "her2k" );
                magma_zher2k( MagmaLower, MagmaNoTrans, pm_old-pn_old, pn_old, c_neg_one,
                     dA(indi_old+pn_old, indj_old), ldda,
                     dW + pn_old,            pm_old, d_one,
                     dA(indi_old+pn_old, indi_old+pn_old), ldda, queues[0] );
                trace_gpu_end( 0, 2 );

                trace_cpu_start( 0, "sync", "sync on 1" );
                magma_queue_sync( queues[1] );
                trace_cpu_end( 0 );
                magma_zq_to_panel(MagmaUpper, pn-1, A(i, i+1), lda, work);
            }

            /* ==========================================================
               QR factorization on a panel starting nb off of the diagonal.
               Prepare the V and T matrices.
               ==========================================================  */
            #ifdef TRACING
            snprintf( buf, sizeof(buf), "panel %lld", (long long) i );
            #endif
            trace_cpu_start( 0, "geqrf", buf );
            lapackf77_zgeqrf(&pm, &pn, A(indi, indj), &lda,
                       tau_ref(i), work, &lwork, info);
            
            /* Form the matrix T */
                        pk=min(pm,pn);
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                          &pm, &pk, A(indi, indj), &lda,
                          tau_ref(i), hT, &nb);

            /* Prepare V - put 0s in the upper triangular part of the panel
               (and 1s on the diagonal), temporaly storing the original in work */
            magma_zpanel_to_q(MagmaUpper, pk, A(indi, indj), lda, work);
            trace_cpu_end( 0 );

            /* Send V from the CPU to the GPU */
            trace_gpu_start( 0, 0, "set", "set V and T" );
            magma_zsetmatrix_async( pm, pk,
                                    A(indi, indj),  lda,
                                    dA(indi, indj), ldda, queues[0] );

            /* Send the triangular factor T to the GPU */
            magma_zsetmatrix_async( pk, pk,
                                    hT,       nb,
                                    dT(i), lddt, queues[0] );
            trace_gpu_end( 0, 0 );
            
            /* ==========================================================
               Compute W:
               1. X = A (V T)
               2. W = X - 0.5* V * (T' * (V' * X))
               ==========================================================  */
            /* dwork = V T */
            trace_cpu_start( 0, "sync", "sync on 0" );
            // this sync is done here to be sure that the copy has been finished
            // because below we made a restore magma_zq_to_panel and this restore need
            // to ensure that the copy has been finished. we did it here to allow
            // overlapp of restore with next gemm and symm.
            magma_queue_sync( queues[0] );
            trace_cpu_end( 0 );
            
            trace_gpu_start( 0, 2, "gemm", "work = V*T" );
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, pm, pk, pk,
                        c_one, dA(indi, indj), ldda,
                        dT(i), lddt,
                        c_zero, dwork, pm, queues[0] );
            trace_gpu_end( 0, 2 );
            
            /* dW = X = A*V*T. dW = A*dwork */
            trace_gpu_start( 0, 2, "hemm", "X = A*work" );
            magma_zhemm( MagmaLeft, uplo, pm, pk,
                        c_one, dA(indi, indi), ldda,
                        dwork, pm,
                        c_zero, dW, pm, queues[0] );
            trace_gpu_end( 0, 2 );
            /* restore the panel */
            magma_zq_to_panel(MagmaUpper, pk, A(indi, indj), lda, work);
            
            /* dwork = V*T already ==> dwork' = T'*V'
             * compute T'*V'*X ==> dwork'*W ==>
             * dwork + pm*nb = ((T' * V') * X) = dwork' * X = dwork' * W */
            trace_gpu_start( 0, 2, "gemm", "work = T'*V'*X" );
            magma_zgemm( MagmaConjTrans, MagmaNoTrans, pk, pk, pm,
                        c_one, dwork, pm,
                        dW, pm,
                        c_zero, dwork + pm*nb, nb, queues[0] );
            trace_gpu_end( 0, 2 );
            
            /* W = X - 0.5 * V * T'*V'*X
             *   = X - 0.5 * V * (dwork + pm*nb) = W - 0.5 * V * (dwork + pm*nb) */
            trace_gpu_start( 0, 2, "gemm", "W = X - 0.5*V*(T'*V'*X)" );
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, pm, pk, pk,
                        c_neg_half, dA(indi, indj), ldda,
                        dwork + pm*nb, nb,
                        c_one,     dW, pm, queues[0] );
            trace_gpu_end( 0, 2 );

            /* ==========================================================
               Update the unreduced submatrix A(i+ib:n,i+ib:n), using
               an update of the form:  A := A - V*W' - W*V'
               ==========================================================  */
            if (i + nb <= n-nb) {
                // There would be next iteration;
                // do lookahead - update the next panel
                trace_gpu_start( 0, 2, "gemm", "gemm 4 next panel left" );
                magma_zgemm( MagmaNoTrans, MagmaConjTrans, pm, pn, pn, c_neg_one,
                            dA(indi, indj), ldda,
                            dW,                 pm, c_one,
                            dA(indi, indi), ldda, queues[0] );
                trace_gpu_end( 0, 2 );
            
                trace_gpu_start( 0, 2, "gemm", "gemm 5 next panel right" );
                magma_zgemm( MagmaNoTrans, MagmaConjTrans, pm, pn, pn, c_neg_one,
                            dW,                 pm,
                            dA(indi, indj), ldda, c_one,
                            dA(indi, indi), ldda, queues[0] );
                trace_gpu_end( 0, 2 );
                magma_event_record(Pupdate_event, queues[0]);
            }
            else {
                /* no look-ahead as this is last iteration */
                trace_gpu_start( 0, 2, "her2k", "her2k last iteration" );
                magma_zher2k( MagmaLower, MagmaNoTrans, pk, pk, c_neg_one,
                             dA(indi, indj), ldda,
                             dW,                 pm, d_one,
                             dA(indi, indi), ldda, queues[0] );
                trace_gpu_end( 0, 2 );
            }
            
            indi_old = indi;
            indj_old = indj;
            pm_old   = pm;
            pn_old   = pn;
        }  // end loop for (i)

        /* Send the last block to the CPU */
        pk = min(pm,pn);
        if (1 <= n-nb) {
            magma_zpanel_to_q(MagmaUpper, pk-1, A(n-pk+1, n-pk+2), lda, work);
            trace_gpu_start( 0, 2, "get", "get last block" );
            magma_zgetmatrix( pk, pk,
                              dA(n-pk+1, n-pk+1), ldda,
                              A(n-pk+1, n-pk+1),  lda, queues[0] );
            trace_gpu_end( 0, 2 );
            magma_zq_to_panel(MagmaUpper, pk-1, A(n-pk+1, n-pk+2), lda, work);
        }
    }// end of LOWER
    
    trace_finalize( "zhetrd_he2hb.svg", "trace.css" );

    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( Pupdate_event );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_free( dA );

    magma_set_lapack_numthreads( orig_threads );

    return *info;
} /* magma_zhetrd_he2hb */
