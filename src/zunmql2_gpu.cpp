/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZUNMQL overwrites the general complex M-by-N matrix C with

    @verbatim
                               SIDE = MagmaLeft   SIDE = MagmaRight
    TRANS = MagmaNoTrans:      Q * C              C * Q
    TRANS = Magma_ConjTrans:   Q**H * C           C * Q**H
    @endverbatim

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by ZGEQLF.
    Q is of order M if SIDE = MagmaLeft
    and  of order N if SIDE = MagmaRight.

    Arguments
    ---------
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

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQLF in the last k columns of its array argument dA.
            The diagonal and the lower part
            are destroyed, the reflectors are not modified.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.
            If SIDE = MagmaLeft,  LDDA >= max(1,M);
            if SIDE = MagmaRight, LDDA >= max(1,N).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQLF.

    @param[in,out]
    dC      COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

    @param[in]
    lddc    INTEGER
            The leading dimension of the array dC. LDDC >= max(1,M).

    @param[in]
    wA      COMPLEX_16 array, dimension
                                 (LDWA,M) if SIDE = MagmaLeft
                                 (LDWA,N) if SIDE = MagmaRight
            The vectors which define the elementary reflectors, as
            returned by ZHETRD_GPU.
            (A copy of the upper or lower part of dA, on the host.)

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.
            If SIDE = MagmaLeft,  LDWA >= max(1,M);
            if SIDE = MagmaRight, LDWA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_unmql
*******************************************************************************/
extern "C" magma_int_t
magma_zunmql2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex    *tau,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    const magmaDoubleComplex *wA, magma_int_t ldwa,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dC(i_,j_) (dC + (i_) + (j_)*lddc)
    #define wA(i_,j_) (wA + (i_) + (j_)*ldwa)
    
    /* Constants */
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    const magma_int_t nbmax = 64;
    
    /* Local variables */
    magmaDoubleComplex_ptr dwork = NULL, dT = NULL;
    magmaDoubleComplex T[ nbmax*nbmax ];
    magma_int_t i, i1, i2, step, ib, lddwork, nb, mi, ni, nq, nq_i, nw;
    magma_queue_t queue = NULL;

    // Parameter adjustments for Fortran indexing
    wA -= 1 + ldwa;
    dC -= 1 + lddc;
    --tau;

    *info  = 0;
    bool left   = (side == MagmaLeft);
    bool notran = (trans == MagmaNoTrans);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }

    /* Test the input arguments */
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
    } else if (ldda < max(1,nq)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (ldwa < max(1,nq)) {
        *info = -12;
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return *info;
    }

    // size of the block
    nb = nbmax;

    lddwork = nw;
    
    /* Use hybrid CPU-GPU code */
    if ( (  left &&   notran) ||
         (! left && ! notran) )
    {
        i1 = 1;
        i2 = k;
        step = nb;
    } else {
        i1 = ((k - 1)/nb)*nb + 1;
        i2 = 1;
        step = -nb;
    }
    
    // silence "uninitialized" warnings
    mi = 0;
    ni = 0;
    
    if (left) {
        ni = n;
    } else {
        mi = m;
    }
    
    // dwork is (n or m) x nb + nb x nb, for left or right respectively
    if (MAGMA_SUCCESS != magma_zmalloc( &dwork, lddwork*nb + nb*nb )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        goto cleanup;
    }
    dT = dwork + lddwork*nb;
    
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    // in bottom k x k portion of dA,
    // set nb-1 sub-diagonals to 0, and diagonal to 1, in 
    // This way we can copy V directly to the GPU,
    // with the lower triangle parts already set to identity.
    // A is nq x k, either m x k (left) or n x k (right)
    magmablas_zlaset_band( MagmaLower, k, k, nb, c_zero, c_one, dA(nq-k,0), ldda, queue );
    
    for (i = i1; (step < 0 ? i >= i2 : i <= i2); i += step) {
        ib = min( nb, k - i + 1 );
        
        /* Form the triangular factor of the block reflector
           H = H(i+ib-1) . . . H(i+1) H(i) */
        nq_i = nq - k + i + ib - 1;
        lapackf77_zlarft( "Backward", "Columnwise", &nq_i, &ib,
                          wA(1,i), &ldwa, &tau[i], T, &ib );
        
        if (left) {
            /* H or H^H is applied to C(1:m-k+i+ib-1,1:n) */
            mi = m - k + i + ib - 1;
        }
        else {
            /* H or H^H is applied to C(1:m,1:n-k+i+ib-1) */
            ni = n - k + i + ib - 1;
        }
        
        /* Apply H or H^H; First copy T to the GPU */
        magma_zsetmatrix( ib, ib, T, ib, dT, ib, queue );
        magma_zlarfb_gpu( side, trans, MagmaBackward, MagmaColumnwise,
                          mi, ni, ib,
                          dA(0,i-1), ldda, dT, ib,  // dA using 0-based indices here
                          dC(1,1), lddc,
                          dwork, lddwork, queue );
    }

cleanup:
    magma_queue_sync( queue );
    magma_queue_destroy( queue );
    magma_free( dwork );

    return *info;
} /* magma_zunmql */
