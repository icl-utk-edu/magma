/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar
       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZUNMQR overwrites the general complex M-by-N matrix C with

    @verbatim
                                SIDE = MagmaLeft    SIDE = MagmaRight
    TRANS = MagmaNoTrans:       Q * C               C * Q
    TRANS = Magma_ConjTrans:    Q**H * C            C * Q**H
    @endverbatim

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = MagmaLeft and of order N
    if SIDE = MagmaRight.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply Q or Q**H from the Left;
      -     = MagmaRight:     apply Q or Q**H from the Right.

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q;
      -     = Magma_ConjTrans: Conjugate transpose, apply Q**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = MagmaLeft,  M >= K >= 0;
            if SIDE = MagmaRight, N >= K >= 0.

    @param[in]
    A       COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.
            If SIDE = MagmaLeft,  LDA >= max(1,M);
            if SIDE = MagmaRight, LDA >= max(1,N).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    @param[in,out]
    C       COMPLEX_16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    @param[in]
    ldc     INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If SIDE = MagmaLeft,  LWORK >= max(1,N);
            if SIDE = MagmaRight, LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = MagmaLeft, and
            LWORK >= M*NB if SIDE = MagmaRight, where NB is the optimal
            blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_unmqr
*******************************************************************************/
extern "C" magma_int_t
magma_zunmqr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *C,    magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info)
{
#define  A(i, j) (A + (j)*lda  + (i))
#define  C(i, j) (C + (j)*ldc  + (i))

#define    dC(gpui,      i, j) (dw[gpui] + (j)*lddc + (i))
#define  dA_c(gpui, ind, i, j) (dw[gpui] + maxnlocal*lddc + (ind)*lddar*lddac + (i) + (j)*lddac)
#define  dA_r(gpui, ind, i, j) (dw[gpui] + maxnlocal*lddc + (ind)*lddar*lddac + (i) + (j)*lddar)
#define    dT(gpui, ind)       (dw[gpui] + maxnlocal*lddc + 2*lddac*lddar + (ind)*((nb+1)*nb))
#define dwork(gpui, ind)       (dw[gpui] + maxnlocal*lddc + 2*lddac*lddar + 2*((nb+1)*nb) + (ind)*(lddwork*nb))

    /* Constants */
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    /* Local variables */
    const char* side_  = lapack_side_const( side );
    const char* trans_ = lapack_trans_const( trans );

    magma_int_t nb = 128;
    magmaDoubleComplex *T = NULL;
    magmaDoubleComplex_ptr dw[MagmaMaxGPUs] = { NULL };
    magma_queue_t queues[MagmaMaxGPUs][2] = {{ NULL }};
#if defined(MAGMA_HAVE_SYCL)
    magma_event_t events[MagmaMaxGPUs][2] = {{ sycl::event() }};
#else
    magma_event_t events[MagmaMaxGPUs][2] = {{ NULL }};
#endif

    magma_int_t ind_c;
    magma_device_t dev;
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    *info = 0;

    magma_int_t left   = (side == MagmaLeft);
    magma_int_t notran = (trans == MagmaNoTrans);
    magma_int_t lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    magma_int_t nq, nw;
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }

    if (! left && side != MagmaRight) {
        *info = -1;
    } else if (! notran && trans != Magma_ConjTrans) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (lda < max(1,nq)) {
        *info = -7;
    } else if (ldc < max(1,m)) {
        *info = -10;
    } else if (lwork < max(1,nw) && ! lquery) {
        *info = -12;
    }

    magma_int_t lwkopt = max(1,nw) * nb;
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
    if (m == 0 || n == 0 || k == 0) {
        work[0] = c_one;
        return *info;
    }

    if (nb >= k) {
        /* Use CPU code */
        lapackf77_zunmqr(side_, trans_, &m, &n, &k, A, &lda, tau,
                         C, &ldc, work, &lwork, info);
        return *info;
    }

    magma_int_t lddc = magma_roundup( m, 64 );  // TODO why 64 instead of 32 ?
    magma_int_t lddac = nq;
    magma_int_t lddar = nb;
    magma_int_t lddwork = nw;

    magma_int_t nlocal[ MagmaMaxGPUs ] = { 0 };

    magma_int_t nb_l=256;
    magma_int_t nbl = magma_ceildiv( n, nb_l ); // number of blocks
    magma_int_t maxnlocal = magma_ceildiv( nbl, ngpu )*nb_l;

    ngpu = min( ngpu, magma_ceildiv( n, nb_l )); // Don't use GPU that will not have data.

    magma_int_t ldw = maxnlocal*lddc // dC
                    + 2*lddac*lddar // 2*dA
                    + 2*(nb + 1 + lddwork)*nb; // 2*(dT and dwork)

    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &T, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto cleanup;
    }
    for (dev = 0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        if (MAGMA_SUCCESS != magma_zmalloc( &dw[dev], ldw )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
        magma_queue_create( dev, &queues[dev][0] );
        magma_queue_create( dev, &queues[dev][1] );
        magma_event_create( &events[dev][0] );
        magma_event_create( &events[dev][1] );
    }

    /* Use hybrid CPU-MGPU code */
    if (left) {
        //copy C to mgpus
        for (magma_int_t i = 0; i < nbl; ++i) {
            dev = i % ngpu;
            magma_setdevice( dev );
            magma_int_t kb = min(nb_l, n-i*nb_l);
            magma_zsetmatrix_async( m, kb,
                                   C(0, i*nb_l), ldc,
                                   dC(dev, 0, i/ngpu*nb_l), lddc, queues[dev][0] );
            nlocal[dev] += kb;
        }

        magma_int_t i1, i2, i3;
        if ( !notran ) {
            i1 = 0;
            i2 = k;
            i3 = nb;
        } else {
            i1 = (k - 1) / nb * nb;
            i2 = 0;
            i3 = -nb;
        }

        ind_c = 0;

        for (magma_int_t i = i1; (i3 < 0 ? i >= i2 : i < i2); i += i3) {
            // start the copy of A panel
            magma_int_t kb = min(nb, k - i);
            for (dev = 0; dev < ngpu; ++dev) {
                magma_setdevice( dev );
                magma_event_sync( events[dev][ind_c] ); // check if the new data can be copied
                magma_zsetmatrix_async(nq-i, kb,
                                       A(i, i),                 lda,
                                       dA_c(dev, ind_c, i, 0), lddac, queues[dev][0] );
                // set upper triangular part of dA to identity
                magmablas_zlaset_band( MagmaUpper, kb, kb, kb, c_zero, c_one, dA_c(dev, ind_c, i, 0), lddac, queues[dev][0] );
            }

            /* Form the triangular factor of the block reflector
             H = H(i) H(i+1) . . . H(i+ib-1) */
            magma_int_t nqi = nq - i;
            lapackf77_zlarft("F", "C", &nqi, &kb, A(i, i), &lda,
                             &tau[i], T, &kb);

            /* H or H' is applied to C(1:m,i:n) */

            /* Apply H or H'; First copy T to the GPU */
            for (dev = 0; dev < ngpu; ++dev) {
                magma_setdevice( dev );
                magma_zsetmatrix_async(kb, kb,
                                       T,               kb,
                                       dT(dev, ind_c), kb, queues[dev][0] );
            }

            for (dev = 0; dev < ngpu; ++dev) {
                magma_setdevice( dev );
                magma_queue_sync( queues[dev][0] ); // check if the data was copied
                magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                                 m-i, nlocal[dev], kb,
                                 dA_c(dev, ind_c, i, 0), lddac, dT(dev, ind_c), kb,
                                 dC(dev, i, 0), lddc,
                                 dwork(dev, ind_c), lddwork, queues[dev][1] );
                magma_event_record(events[dev][ind_c], queues[dev][1] );
            }

            ind_c = (ind_c+1)%2;
        }

        for (dev = 0; dev < ngpu; ++dev) {
            magma_setdevice( dev );
            magma_queue_sync( queues[dev][1] );
        }

        //copy C from mgpus
        for (magma_int_t i = 0; i < nbl; ++i) {
            dev = i % ngpu;
            magma_setdevice( dev );
            magma_int_t kb = min(nb_l, n-i*nb_l);
            magma_zgetmatrix( m, kb,
                              dC(dev, 0, i/ngpu*nb_l), lddc,
                              C(0, i*nb_l), ldc, queues[dev][1] );
//            magma_zgetmatrix_async( m, kb,
//                                   dC(dev, 0, i/ngpu*nb_l), lddc,
//                                   C(0, i*nb_l), ldc, queues[dev][0] );
        }
    } else {
        *info = MAGMA_ERR_NOT_IMPLEMENTED;
        magma_xerbla( __func__, -(*info) );
        goto cleanup;
        
        /*
        if ( notran ) {
            i1 = 0;
            i2 = k;
            i3 = nb;
        } else {
            i1 = (k - 1) / nb * nb;
            i2 = 0;
            i3 = -nb;
        }

        mi = m;
        ic = 0;

        for (i = i1; (i3 < 0 ? i >= i2 : i < i2); i += i3) {
            ib = min(nb, k - i);
            
            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1)
            i__4 = nq - i;
            lapackf77_zlarft("F", "C", &i__4, &ib, A(i, i), &lda,
            &tau[i], T, &ib);
            
            // 1) copy the panel from A to the GPU, and
            // 2) set upper triangular part of dA to identity
            magma_zsetmatrix( i__4, ib, A(i, i), lda, dA(i, 0), ldda, queues[dev][1] );
            magmablas_zlaset_band( MagmaUpper, ib, ib, ib, c_zero, c_one, dA(i, 0), ldda, queues[dev][1] );
            
            // H or H' is applied to C(1:m,i:n)
            ni = n - i;
            jc = i;
            
            // Apply H or H'; First copy T to the GPU
            magma_zsetmatrix( ib, ib, T, ib, dT, ib, queues[dev][1] );
            magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
            mi, ni, ib,
            dA(i, 0), ldda, dT, ib,
            dC(ic, jc), lddc,
            dwork, lddwork, queues[dev][1] );
        }
        */
    }

cleanup:
    work[0] = magma_zmake_lwork( lwkopt );

    for (dev = 0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        magma_event_destroy( events[dev][0] );
        magma_event_destroy( events[dev][1] );
        magma_queue_destroy( queues[dev][0] );
        magma_queue_destroy( queues[dev][1] );
        magma_free( dw[dev] );
    }
    magma_setdevice( orig_dev );
    magma_free_pinned( T );

    return *info;
} /* magma_zunmqr */
