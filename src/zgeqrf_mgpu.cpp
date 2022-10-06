/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R. This is a GPU interface of the routine.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dlA     COMPLEX_16 array of pointers on the GPU, dimension (ngpu).
            On entry, the M-by-N matrix A distributed over GPUs
            (d_lA[d] points to the local matrix on d-th GPU).
            It uses 1D block column cyclic format with the block size of nb,
            and each local matrix is stored by column.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dlA[], magma_int_t ldda,
    magmaDoubleComplex *tau,
    magma_int_t *info )
{
    #define dlA(dev, i, j)   (dlA[dev] + (i) + (j)*(ldda))
    #define hpanel(i)        (hpanel + (i))

    // set to NULL to make cleanup easy: free(NULL) does nothing.
    magmaDoubleComplex *dwork[MagmaMaxGPUs]={NULL}, *dpanel[MagmaMaxGPUs]={NULL};
    magmaDoubleComplex *hwork=NULL, *hpanel=NULL;
    magma_queue_t queues[MagmaMaxGPUs][2]={{NULL}};
#if defined(MAGMA_HAVE_SYCL)
    magma_event_t panel_event[MagmaMaxGPUs]={sycl::event()};
#else
    magma_event_t panel_event[MagmaMaxGPUs]={NULL};
#endif

    magma_int_t i, j, min_mn, dev, ldhpanel, lddwork, rows;
    magma_int_t ib, nb;
    magma_int_t lhwork, lwork;
    magma_int_t panel_dev, i_local, i_nb_local, n_local[MagmaMaxGPUs], la_dev, dpanel_offset;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    min_mn = min(m,n);
    if (min_mn == 0)
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    nb = magma_get_zgeqrf_nb( m, n );

    /* dwork is (n*nb) --- for T (nb*nb) and zlarfb work ((n-nb)*nb) ---
     *        + dpanel (ldda*nb), on each GPU.
     * I think zlarfb work could be smaller, max(n_local[:]).
     * Oddly, T and zlarfb work get stacked on top of each other, both with lddwork=n.
     * on GPU that owns panel, set dpanel = dlA(dev,i,i_local).
     * on other GPUs,          set dpanel = dwork[dev] + dpanel_offset. */
    lddwork = n;
    dpanel_offset = lddwork*nb;
    for( dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        if ( MAGMA_SUCCESS != magma_zmalloc( &(dwork[dev]), (lddwork + ldda)*nb )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto CLEANUP;
        }
    }

    /* hwork is MAX( workspace for zgeqrf (n*nb), two copies of T (2*nb*nb) )
     *        + hpanel (m*nb).
     * for last block, need 2*n*nb total. */
    ldhpanel = m;
    lhwork = max( n*nb, 2*nb*nb );
    lwork = max( lhwork + ldhpanel*nb, 2*n*nb );
    if ( MAGMA_SUCCESS != magma_zmalloc_pinned( &hwork, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto CLEANUP;
    }
    hpanel = hwork + lhwork;

    /* Set the number of local n for each GPU */
    for( dev=0; dev < ngpu; dev++ ) {
        n_local[dev] = ((n/nb)/ngpu)*nb;
        if (dev < (n/nb) % ngpu)
            n_local[dev] += nb;
        else if (dev == (n/nb) % ngpu)
            n_local[dev] += n % nb;
    }

    for( dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev][0] );
        magma_queue_create( dev, &queues[dev][1] );
        magma_event_create( &panel_event[dev] );
    }

    if ( nb < min_mn ) {
        /* Use blocked code initially */
        // Note: as written, ib cannot be < nb.
        for( i = 0; i < min_mn-nb; i += nb ) {
            /* Set the GPU number that holds the current panel */
            panel_dev = (i/nb) % ngpu;
            
            /* Set the local index where the current panel is (j == i) */
            i_local = i/(nb*ngpu)*nb;
            
            ib = min(min_mn-i, nb);
            rows = m-i;
            
            /* Send current panel to the CPU, after panel_event indicates it has been updated */
            magma_setdevice( panel_dev );
            magma_queue_wait_event( queues[panel_dev][1], panel_event[panel_dev] );
            magma_zgetmatrix_async( rows, ib,
                                    dlA(panel_dev, i, i_local), ldda,
                                    hpanel(i),                  ldhpanel, 
                                    queues[panel_dev][1] );
            magma_queue_sync( queues[panel_dev][1] );

            // Factor panel
            lapackf77_zgeqrf( &rows, &ib, hpanel(i), &ldhpanel, tau+i,
                              hwork, &lhwork, info );
            if ( *info != 0 ) {
                fprintf( stderr, "error %lld\n", (long long) *info );
            }

            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1)
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              hpanel(i), &ldhpanel, tau+i, hwork, &ib );

            magma_zpanel_to_q( MagmaUpper, ib, hpanel(i), ldhpanel, hwork + ib*ib );
            // Send the current panel back to the GPUs
            for( dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                if (dev == panel_dev)
                    dpanel[dev] = dlA(dev, i, i_local);
                else
                    dpanel[dev] = dwork[dev] + dpanel_offset;
                magma_zsetmatrix_async( rows, ib,
                                        hpanel(i),   ldhpanel,
                                        dpanel[dev], ldda, 
                                        queues[dev][0] );
            }
            for( dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_queue_sync( queues[dev][0] );
            }

            // TODO: if magma_zpanel_to_q copied whole block, wouldn't need to restore
            // -- just send the copy to the GPUs.
            // TODO: also, could zero out the lower triangle and use Azzam's larfb w/ gemm.
            
            /* Restore the panel */
            magma_zq_to_panel( MagmaUpper, ib, hpanel(i), ldhpanel, hwork + ib*ib );

            if (i + ib < n) {
                /* Send the T matrix to the GPU. */
                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    magma_zsetmatrix_async( ib, ib,
                                            hwork,      ib,
                                            dwork[dev], lddwork, 
                                            queues[dev][0] );
                }
                
                la_dev = (panel_dev+1) % ngpu;
                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    if (dev == la_dev && i+nb < min_mn-nb) {
                        // If not last panel,
                        // for look-ahead panel, apply H' to A(i:m,i+ib:i+2*ib)
                        i_nb_local = (i+nb)/(nb*ngpu)*nb;
                        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                          rows, ib, ib,
                                          dpanel[dev],             ldda,       // V
                                          dwork[dev],              lddwork,    // T
                                          dlA(dev, i, i_nb_local), ldda,       // C
                                          dwork[dev]+ib,           lddwork,    // work
                                          queues[dev][0] );  
                        magma_event_record( panel_event[dev], queues[dev][0] );
                        // for trailing matrix, apply H' to A(i:m,i+2*ib:n)
                        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                          rows, n_local[dev]-(i_nb_local+ib), ib,
                                          dpanel[dev],                ldda,       // V
                                          dwork[dev],                 lddwork,    // T
                                          dlA(dev, i, i_nb_local+ib), ldda,       // C
                                          dwork[dev]+ib,              lddwork,    // work
                                          queues[dev][0] ); 
                    }
                    else {
                        // for trailing matrix, apply H' to A(i:m,i+ib:n)
                        i_nb_local = i_local;
                        if (dev <= panel_dev) {
                            i_nb_local += ib;
                        }
                        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                          rows, n_local[dev]-i_nb_local, ib,
                                          dpanel[dev],             ldda,       // V
                                          dwork[dev],              lddwork,    // T
                                          dlA(dev, i, i_nb_local), ldda,       // C
                                          dwork[dev]+ib,           lddwork,    // work
                                          queues[dev][0] );
                    }
                }
                // Restore top of panel (after larfb is done)
                magma_setdevice( panel_dev );
                magma_zsetmatrix_async( ib, ib,
                                        hpanel(i),                  ldhpanel,
                                        dlA(panel_dev, i, i_local), ldda, 
                                        queues[panel_dev][0] );
            }
        }
    }
    else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block row. */
    if (i < min_mn) {
        rows = m-i;
        for( j=i; j < n; j += nb ) {
            panel_dev = (j/nb) % ngpu;
            i_local = j/(nb*ngpu)*nb;
            ib = min( n-j, nb );
            magma_setdevice( panel_dev );
            magma_zgetmatrix( rows, ib,
                              dlA(panel_dev, i, i_local), ldda,
                              hwork + (j-i)*rows,         rows,
                              queues[panel_dev][0] );
        }

        // needs lwork >= 2*n*nb:
        // needs (m-i)*(n-i) for last block row, bounded by nb*n.
        // needs (n-i)*nb    for zgeqrf work,    bounded by n*nb.
        ib = n-i;  // total columns in block row
        lhwork = lwork - ib*rows;
        lapackf77_zgeqrf( &rows, &ib, hwork, &rows, tau+i, hwork + ib*rows, &lhwork, info );
        if ( *info != 0 ) {
            fprintf( stderr, "error %lld\n", (long long) *info );
        }
        
        for( j=i; j < n; j += nb ) {
            panel_dev = (j/nb) % ngpu;
            i_local = j/(nb*ngpu)*nb;
            ib = min( n-j, nb );
            magma_setdevice( panel_dev );
            magma_zsetmatrix( rows, ib,
                              hwork + (j-i)*rows,         rows,
                              dlA(panel_dev, i, i_local), ldda,
                              queues[panel_dev][0] );
        }
    }

CLEANUP:
    // free(NULL) does nothing.
    for( dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_destroy( queues[dev][0]   );
        magma_queue_destroy( queues[dev][1]   );
        magma_event_destroy( panel_event[dev] );
        magma_free( dwork[dev] );
    }
    magma_free_pinned( hwork );
    magma_setdevice( orig_dev );

    return *info;
} /* magma_zgeqrf2_mgpu */
