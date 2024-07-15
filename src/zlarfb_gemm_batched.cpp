/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
       @precisions normal z -> s d c
*/
#include "magma_internal.h"

extern "C" magma_int_t
magma_zlarfb_gemm_internal_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV_array[],    magma_int_t vi, magma_int_t vj, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT_array[],    magma_int_t Ti, magma_int_t Tj, magma_int_t lddt,
    magmaDoubleComplex_ptr dC_array[],          magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magmaDoubleComplex_ptr dwork_array[],       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue)
{
    // Constants
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    /* Function Body */
    magma_int_t info = 0;
    if (m <= 0 || n <= 0) {
        return info;
    }

    // Local variables
    magma_int_t ldwvt = (m > n ?  k : m);
    magma_int_t ldw;
    if ( side == MagmaLeft ) {
        ldw = k;
    } else {
        ldw = m;
    }

    // opposite of trans
    magma_trans_t transt;
    if (trans == MagmaNoTrans)
        transt = Magma_ConjTrans;
    else
        transt = MagmaNoTrans;

    // TODO: is this a bug that it isn't used?
    MAGMA_UNUSED( transt );

    // whether V is stored transposed or not
    magma_trans_t notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = MagmaNoTrans;
        transV   = Magma_ConjTrans;
    }
    else {
        notransV = Magma_ConjTrans;
        transV   = MagmaNoTrans;
    }

    if ( side == MagmaLeft ) {
        // Form H C or H^H C
        // Comments assume H C.
        // When forming H^H C, T gets transposed via transt for m >= n or by trans for m < n.

        // W = V^H C
        magma_zgemm_batched_core( Magma_ConjTrans,notransV, /*NontransLeft*/
                     k, n, m,
                     c_one,  dV_array,    vi, vj, lddv,
                             dC_array,    Ci, Cj, lddc,
                     c_zero, dwork_array,  0,  0, ldw,
                     batchCount, queue );

        if (m <= n) {
            // W2 = V T
            magma_zgemm_batched_core( notransV, trans, /* (NoTrans), trans(ConjTrans),*/
                         m, k, k,
                         c_one,  dV_array,     vi, vj, lddv,
                                 dT_array,     Ti, Tj, lddt,
                         c_zero, dworkvt_array, 0,  0, ldwvt,
                         batchCount, queue );

            // C = C - W2 W = C - V T V^H C = (I - V T V^H) C = H C
            magma_zgemm_batched_core( MagmaNoTrans, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dworkvt_array, 0,  0, ldwvt,
                                    dwork_array,   0,  0, ldw,
                         c_one,     dC_array,     Ci, Cj, lddc,
                         batchCount, queue );
        }
        else {
            // W2 = T W  = T  V^H C
            magma_zgemm_batched_core( trans, MagmaNoTrans,
                         k, n, k,
                         c_one,  dT_array,     Ti, Tj, lddt,
                                 dwork_array,   0,  0, ldw,
                         c_zero, dworkvt_array, 0,  0, ldwvt,
                         batchCount, queue );

            // C = C - V W2 = C - V T V^H C = (I - V T V^H) C = H C
            magma_zgemm_batched_core( notransV, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dV_array,     vi, vj,  lddv,
                                    dworkvt_array, 0,  0,  ldwvt,
                         c_one,     dC_array,      Ci, Cj, lddc,
                         batchCount, queue );
        }
    }
    else {
        // Form C H or C H^H
        // Comments assume C H.
        // When forming C H^H, T gets transposed via trans.

        // W = C V
        magma_zgemm_batched_core( MagmaNoTrans, notransV,
                     m, k, n,
                     c_one,  dC_array,    Ci, Cj, lddc,
                             dV_array,    vi, vj, lddv,
                     c_zero, dwork_array,  0,  0, ldw,
                     batchCount, queue );

        if (m <= n) {
            // W2 = W T = C V T
            magma_zgemm_batched_core( MagmaNoTrans, trans,
                         m, k, k,
                         c_one,  dwork_array,    0,  0, ldw,
                                 dT_array,      Ti, Tj, lddt,
                         c_zero, dworkvt_array,  0,  0, ldwvt,
                         batchCount, queue );

            // C = C - W2 V^H = C - C V T V^H = C (I - V T V^H) = C H
            magma_zgemm_batched_core( MagmaNoTrans, transV,
                         m, n, k,
                         c_neg_one, dworkvt_array, 0,  0, ldwvt,
                                    dV_array,     vi, vj, lddv,
                         c_one,     dC_array,     Ci, Cj, lddc,
                         batchCount, queue );
        }
        else {
            // W2 = T V^H
            magma_zgemm_batched_core( trans, transV,
                         k, n, k,
                         c_one,  dT_array, Ti, Tj, lddt,
                                 dV_array, vi, vj, lddv,
                         c_zero, dworkvt_array, 0, 0, ldwvt,
                         batchCount, queue );

            // C = C - W W2 = C - C V T V^H = C (I - V T V^H) = C H
            magma_zgemm_batched_core( MagmaNoTrans, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dwork_array,    0,  0, ldw,
                                    dworkvt_array,  0,  0, ldwvt,
                         c_one,     dC_array,      Ci, Cj, lddc,
                         batchCount, queue );
        }
    }

    return info;
} /* magma_zlarfb_internal */

/***************************************************************************//**
    Purpose
    -------
    ZLARFB applies a complex block reflector H or its transpose H^H to a
    COMPLEX_16 m by n matrix C, from the left.

    __Note that this function assumes__ that the upper part of dV_array is 0
    because it is referenced. Same for upper/lower part of dT_array.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply H or H^H from the Left
      -     = MagmaRight:     apply H or H^H from the Right

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    apply H   (No transpose)
      -     = Magma_ConjTrans: apply H^H (Conjugate transpose)

    @param[in]
    direct  magma_direct_t
            Indicates how H is formed from a product of elementary
            reflectors
      -     = MagmaForward:  H = H(1) H(2) . . . H(k) (Forward)
      -     = MagmaBackward: H = H(k) . . . H(2) H(1) (Backward)

    @param[in]
    storev  magma_storev_t
            Indicates how the vectors which define the elementary
            reflectors are stored:
      -     = MagmaColumnwise: Columnwise
      -     = MagmaRowwise:    Rowwise

    @param[in]
    m       INTEGER
            The number of rows of the matrix C.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C.

    @param[in]
    k       INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    @param[in]
    dV_array      COMPLEX_16 array on the GPU, dimension
                (LDDV,K) if STOREV = MagmaColumnwise
                (LDDV,M) if STOREV = MagmaRowwise and SIDE = MagmaLeft
                (LDDV,N) if STOREV = MagmaRowwise and SIDE = MagmaRight
            The matrix V. See further details.

    @param[in]
    lddv    INTEGER
            The leading dimension of the array V.
            If STOREV = MagmaColumnwise and SIDE = MagmaLeft, LDDV >= max(1,M);
            if STOREV = MagmaColumnwise and SIDE = MagmaRight, LDDV >= max(1,N);
            if STOREV = MagmaRowwise, LDDV >= K.

    @param[in]
    dT_array      COMPLEX_16 array on the GPU, dimension (LDDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    @param[in]
    lddt    INTEGER
            The leading dimension of the array T. LDDT >= K.

    @param[in,out]
    dC_array      COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C, or H^H*C, or C*H, or C*H^H.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDDC >= max(1,M).

    @param
    dwork_array   (workspace) COMPLEX_16 array, dimension (LDWORK,K)

    @param[in]
    ldwork  INTEGER
            The leading dimension of the array WORK.
            If SIDE = MagmaLeft,  LDWORK >= max(1,N);
            if SIDE = MagmaRight, LDWORK >= max(1,M);

    @param
    dworkvt_array (workspace) COMPLEX_16 array, dimension (LDWORKT,K)

    @param[in]
    ldworkvt INTEGER
            The leading dimension of the array WORKVT.
            LDWORKVT >= max(1,min(M,N));

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The shape of the matrix V and the storage of the vectors which define
    the H(i) is best illustrated by the following example with n = 5 and
    k = 3.
    All elements including 0's and 1's are stored, unlike LAPACK.

        DIRECT = MagmaForward and         DIRECT = MagmaForward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = (  1  0  0 )                 V = (  1 v1 v1 v1 v1 )
                     ( v1  1  0 )                     (  0  1 v2 v2 v2 )
                     ( v1 v2  1 )                     (  0  0  1 v3 v3 )
                     ( v1 v2 v3 )
                     ( v1 v2 v3 )

        DIRECT = MagmaBackward and        DIRECT = MagmaBackward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1  0  0 )
                     ( v1 v2 v3 )                     ( v2 v2 v2  1  0 )
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
                     (  0  1 v3 )
                     (  0  0  1 )

    @ingroup magma_larfb_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zlarfb_gemm_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV_array[],    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT_array[],    magma_int_t lddt,
    magmaDoubleComplex_ptr dC_array[],          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork_array[],       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_zlarfb_gemm_internal_batched(
        side, trans, direct, storev,
        m, n, k,
        dV_array, 0, 0, lddv,
        dT_array, 0, 0, lddt,
        dC_array, 0, 0, lddc,
        dwork_array, ldwork,
        dworkvt_array,     ldworkvt,
        batchCount, queue);

        return 0;
} /* magma_zlarfb */
