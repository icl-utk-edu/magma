/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZUNMLQ_GPU overwrites the general complex M-by-N matrix C with

    @verbatim
                                 SIDE = MagmaLeft    SIDE = MagmaRight
    TRANS = MagmaNoTrans:        Q * C               C * Q
    TRANS = MagmaConjTrans:      Q**H * C            C * Q**H
    @endverbatim

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k)**H . . . H(2)**H H(1)**H

    as returned by ZGELQF. Q is of order M if SIDE = MagmaLeft and of order N
    if SIDE = MagmaRight.

    This is the GPU version where dA and dC reside on the device.
    Internally, reflector panels are transferred to the host to form
    block-reflector T factors, which are then uploaded and applied via
    magma_zlarfb_gpu.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:   apply Q or Q**H from the Left;
      -     = MagmaRight:  apply Q or Q**H from the Right.

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q;
      -     = MagmaConjTrans:  Conjugate transpose, apply Q**H.

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
    dA      COMPLEX_16 array on the GPU, dimension
                (LDDA,M) if SIDE = MagmaLeft,
                (LDDA,N) if SIDE = MagmaRight.
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGELQF_GPU in the first k rows of its array argument dA.
            dA is modified by the routine but restored on exit.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA. LDDA >= max(1,K).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGELQF_GPU.

    @param[in,out]
    dC      COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

    @param[in]
    lddc    INTEGER
            The leading dimension of the array dC. LDDC >= max(1,M).

    @param[out]
    hwork   (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, HWORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array HWORK.
            LWORK >= NB * NQ, where NQ = M if SIDE = MagmaLeft, N if SIDE = MagmaRight,
            and NB is the block size returned by magma_get_zgelqf_nb(M,N).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_unmlq
*******************************************************************************/
extern "C" magma_int_t
magma_zunmlq_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex const   *tau,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magmaDoubleComplex       *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dC(i_,j_) (dC + (i_) + (j_)*lddc)
    #define dV(i_,j_) (dV + (i_) + (j_)*ib)
    #define dT(i_,j_) (dT + (i_) + (j_)*ib)
    #define dwork(i_) (dwork + (i_))

    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    magmaDoubleComplex *T, *T2;
    magma_int_t i, i1, i2, ib, ic, jc, nb, mi, ni, nq, nq_i, nw, step;
    magma_int_t lwkopt;
    magma_trans_t transt;

    *info = 0;
    bool left   = (side  == MagmaLeft);
    bool notran = (trans == MagmaNoTrans);
    bool lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }

    nb = magma_get_zgelqf_nb( m, n );
    lwkopt = nb * nq;
    hwork[0] = magma_zmake_lwork( lwkopt );

    if ( ! left && side != MagmaRight ) {
        *info = -1;
    } else if ( ! notran && trans != MagmaConjTrans ) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (ldda < max(1,k)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (lwork < lwkopt && ! lquery) {
        *info = -12;
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
        hwork[0] = c_one;
        return *info;
    }

    magma_int_t ldwork = nw;

    /* Allocate work space on the GPU.
     * nw*nb  for dwork (workspace for zlarfb_gpu)
     * nb*nq  for dV    (reflector panel)
     * nb*nb  for dT    (block reflector factor)
     */
    magmaDoubleComplex_ptr dwork, dV, dT;
    magma_zmalloc( &dwork, nw*nb + nb*nq + nb*nb );
    if ( dwork == NULL ) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dV = dwork + nw*nb;
    dT = dV    + nb*nq;

    /* work space on CPU.
     * nb*nb for T
     * nb*nb for T2, used to save and restore diagonal block of panel  */
    magma_zmalloc_cpu( &T, 2*nb*nb );
    if ( T == NULL ) {
        magma_free( dwork );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    T2 = T + nb*nb;

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if ( (left && notran) || (! left && ! notran) ) {
        i1 = 0;
        i2 = k;
        step = nb;
    } else {
        i1 = ((k - 1) / nb)*nb;
        i2 = 0;
        step = -nb;
    }

    // silence "uninitialized" warnings
    mi = 0;
    ni = 0;

    if (left) {
        ni = n;
        jc = 0;
    } else {
        mi = m;
        ic = 0;
    }

    if (notran) {
        transt = MagmaConjTrans;
    } else {
        transt = MagmaNoTrans;
    }

    for (i = i1; (step < 0 ? i >= i2 : i < i2); i += step) {
        ib = min(nb, k - i);

        /* Download the reflector panel from GPU to CPU (ib rows, nq-i cols) */
        nq_i = nq - i;
        magma_zgetmatrix( ib, nq_i, dA(i, i), ldda, hwork, ib, queue );

        /* Form the triangular factor of the block reflector
           H = H(i) H(i + 1) . . . H(i + ib-1) */
        lapackf77_zlarft( "Forward", "Rowwise", &nq_i, &ib,
                          hwork, &ib, &tau[i], T, &ib );

        /* 1) set lower triangle of panel in hwork to identity,
           2) copy the panel from CPU to the GPU, and
           3) restore hwork                                      */
        magma_zpanel_to_q( MagmaLower, ib, hwork, ib, T2 );
        magma_zsetmatrix( ib, nq_i, hwork, ib, dV(0,0), ib, queue );
        magma_zq_to_panel( MagmaLower, ib, hwork, ib, T2 );

        if (left) {
            /* H or H**H is applied to C(i:m,1:n) */
            mi = m - i;
            ic = i;
        }
        else {
            /* H or H**H is applied to C(1:m,i:n) */
            ni = n - i;
            jc = i;
        }

        /* Apply H or H**H; First copy T to the GPU */
        magma_zsetmatrix( ib, ib, T, ib, dT(0,0), ib, queue );
        magma_zlarfb_gpu( side, transt, MagmaForward, MagmaRowwise,
                          mi, ni, ib,
                          dV(0,0), ib,
                          dT(0,0), ib,
                          dC(ic,jc), lddc,
                          dwork(0), ldwork, queue );
    }

    magma_queue_sync( queue );
    magma_queue_destroy( queue );
    magma_free( dwork );
    magma_free_cpu( T );

    hwork[0] = magma_zmake_lwork( lwkopt );

    return *info;
} /* magma_zunmlq_gpu */
