/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define PRECISION_z

/***************************************************************************//**
    Purpose
    -------
    ZUNMQR overwrites the general complex M-by-N matrix C with

    @verbatim
                               SIDE = MagmaLeft    SIDE = MagmaRight
    TRANS = MagmaNoTrans:      Q * C               C * Q
    TRANS = Magma_ConjTrans:   Q**H * C            C * Q**H
    @endverbatim

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = MagmaLeft and of order N
    if SIDE = MagmaRight.

    - Only SIDE = MagmaLeft is currently supported
    - This is the batch version of the routine

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:   apply Q or Q**H from the Left;
      -     = MagmaRight:  apply Q or Q**H from the Right (not currently supported).

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
    dA_array    Array of pointers, dimension (batchCount)
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument dA.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dA.
            If SIDE = MagmaLeft,  LDDA >= max(1,M);
            if SIDE = MagmaRight, LDDA >= max(1,N).

    @param[in]
    dtau_array    Array of pointers, dimension(batchCount)
            Each is a COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    @param[in,out]
    dC_array    Array of pointers, dimension (batchCount)
            Each is a COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

    @param[in]
    lddc    INTEGER
            The leading dimension of each array DC. LDDC >= max(1,M).

    @param[in,out]
    device_work  Workspace, allocated on device (GPU) memory.

    @param[in,out]
    lwork_device   INTEGER pointer
                   The size of the workspace (device_work) in bytes
                   - lwork_device[0] < 0: a workspace query is assumed, the routine
                     calculates the required amount of workspace and returns
                     it in lwork_device. The workspace itself is not referenced, and no
                     computation is performed.
                   - lwork_device[0] >= 0: the routine assumes that the user has provided
                     a workspace with the size in lwork_device.

    @param[out]
    dinfo_array    INTEGER array on GPU memory. Each entry is either,
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_unmqr
*******************************************************************************/
extern "C" magma_int_t
magma_zunmqr_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dtau_array,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    void *device_work, int64_t *lwork_device,
    magma_int_t *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue)
{
#define dA_array(i,j) dA_array, i, j
#define dC_array(i,j) dC_array, i, j
#define dR_array(i,j) dR_array, i, j
#define dT_array(i,j) dT_array, i, j
#define dtau_array(i) dtau_array, i

    magma_int_t arginfo = 0, minmn, Qn, nb, lddr, lddt, lddw, lddwvt;

    // nb is limited to 32 because of limitations in batch larft
    nb    = 32;

    // lookup recommendation for using zunm2r (from the batch qr factorization tuning)
    magma_int_t use_fused_update = magma_use_zgeqrf_batched_fused_update(m, n, batchCount);

    // for zunm2r, check if fused routines could be launched
    magma_int_t zunm2r_reg_info     = -1;
    magma_int_t zunm2r_sm_info      = -1;
    magma_int_t zunm2r_check_launch =  1;
    magma_int_t zunm2r_reg_nb = 0;
    magma_int_t zunm2r_sm_nb  = 0;
    if(use_fused_update == 1) {
        for(magma_int_t inb = 8; inb > 1; inb /= 2) {
            zunm2r_reg_info = magma_zunm2r_reg_batched(
                                side, trans, m, n, inb, k,
                                dC_array, 0, 0, lddc,
                                dA_array, 0, 0, ldda, dtau_array,  0,
                                zunm2r_check_launch, batchCount, queue );

            if(zunm2r_reg_info == 0) {
                zunm2r_reg_nb = inb;
                break;
            }
        }

        for(magma_int_t inb = 8; inb > 1; inb /= 2) {
            zunm2r_sm_info = magma_zunm2r_sm_batched(
                                side, trans, m, n, inb, k,
                                dC_array, 0, 0, lddc,
                                dA_array, 0, 0, ldda, dtau_array,  0,
                                zunm2r_check_launch, batchCount, queue );

            if(zunm2r_sm_info == 0) {
                zunm2r_sm_nb = inb;
                break;
            }
        }
    }

    // set info to zero
    minmn = min(m, n);
    lddt  = nb;
    lddr  = nb;
    Qn    = (side == MagmaLeft) ? m : n;

    lddw   = (side == MagmaLeft) ? n : m;
    lddwvt = minmn;

    // we will use the same workspace for larft and one of two workspaces in larfb
    // larft         : requires lddt*k per matrix
    // larfb (1st ws): requires lddw*k per matrix
    // so set lddw = max(lddw, lddt)
    lddw = max(lddw, lddt);

    // calculate required workspace
    magma_int_t alignment_bytes = 128;
    magma_int_t alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    magma_int_t alignment_ptr   = alignment_bytes / sizeof(magmaDoubleComplex*);

    size_t ws_unmqr_R   = magma_roundup( batchCount * lddr * nb, alignment);
    size_t ws_unmqr_T   = magma_roundup( batchCount * lddt * nb, alignment);
    size_t ws_larfb_W   = magma_roundup( batchCount * lddw *  k, alignment);    // also used as ws_larft_W
    size_t ws_larfb_Wvt = magma_roundup( batchCount * lddwvt *  k, alignment);

    size_t ws_dR_array   = magma_roundup( batchCount, alignment_ptr );    // dR_array   (for unmqr/ormqr to preserve R)
    size_t ws_dT_array   = magma_roundup( batchCount, alignment_ptr );    // dT_array   (for unmqr to store T)
    size_t ws_dW_array   = magma_roundup( batchCount, alignment_ptr );    // dW_array   (for larft & larfb -- dual use)
    size_t ws_dWvt_array = magma_roundup( batchCount, alignment_ptr );    // dWvt_array (for larfb only)

    // dW_array will account for both larft & larfb, so we should take
    // the maximum of ws_larft_W & ws_larfb_W

    size_t workspace_bytes =  0;
    workspace_bytes += sizeof(magmaDoubleComplex) * ws_unmqr_R;
    workspace_bytes += sizeof(magmaDoubleComplex) * ws_unmqr_T;
    workspace_bytes += sizeof(magmaDoubleComplex) * ws_larfb_W;
    workspace_bytes += sizeof(magmaDoubleComplex) * ws_larfb_Wvt;

    workspace_bytes += sizeof(magmaDoubleComplex*) * ws_dR_array;
    workspace_bytes += sizeof(magmaDoubleComplex*) * ws_dT_array;
    workspace_bytes += sizeof(magmaDoubleComplex*) * ws_dW_array;
    workspace_bytes += sizeof(magmaDoubleComplex*) * ws_dWvt_array;

    // if we can use fused zunm2r, no workspace is required
    workspace_bytes = (zunm2r_reg_info == 0 || zunm2r_sm_info == 0) ? 0 : workspace_bytes;

    // check for workspace query
    if( lwork_device[0] < 0 ) {
        lwork_device[0] = (int64_t)workspace_bytes;
        return arginfo;
    }

    if ( side != MagmaLeft ) {
        printf("Error in %s: only side = MagmaLeft is supported\n", __func__);
        arginfo = -1;
    } else if ( trans != MagmaNoTrans && trans != Magma_ConjTrans ) {
        arginfo = -2;
    } else if (m < 0) {
        arginfo = -3;
    } else if (n < 0) {
        arginfo = -4;
    } else if (k < 0 || k > Qn) {
        arginfo = -5;
    } else if (ldda < max(1,Qn)) {
        arginfo = -7;
    } else if (lddc < max(1,m)) {
        arginfo = -10;
    } else if (lwork_device[0] < (int64_t)(workspace_bytes)) {
        arginfo = -12;
    } else if (batchCount < 0) {
        arginfo = -14;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0 || batchCount == 0) {
        return arginfo;
    }

    magma_memset_async(dinfo_array,  0, batchCount * sizeof(magma_int_t), queue);


    // try fused impl. first
    if(zunm2r_reg_info == 0) {
        zunm2r_reg_info = magma_zunm2r_reg_batched(
                            side, trans, m, n, zunm2r_reg_nb, k,
                            dC_array, 0, 0, lddc,
                            dA_array, 0, 0, ldda, dtau_array,  0,
                            0, batchCount, queue );

        if(zunm2r_reg_info == 0) return arginfo;
    }

    if(zunm2r_sm_info == 0) {
        zunm2r_sm_info = magma_zunm2r_sm_batched(
                            side, trans, m, n, zunm2r_sm_nb, k,
                            dC_array, 0, 0, lddc,
                            dA_array, 0, 0, ldda, dtau_array,  0,
                            0, batchCount, queue );
        if(zunm2r_sm_info == 0) return arginfo;
    }

    // if we are here, then zunm2r did not work, run blocked code
    // split workspace
    magmaDoubleComplex** dR_array   = (magmaDoubleComplex**)(device_work);
    magmaDoubleComplex** dT_array   = dR_array + ws_dR_array;
    magmaDoubleComplex** dW_array   = dT_array + ws_dT_array;
    magmaDoubleComplex** dWvt_array = dW_array + ws_dW_array;

    magmaDoubleComplex *dR   = (magmaDoubleComplex*)(dWvt_array + ws_dWvt_array);
    magmaDoubleComplex *dT   = dR + ws_unmqr_R;
    magmaDoubleComplex *dW   = dT + ws_unmqr_T;
    magmaDoubleComplex *dWvt = dW + ws_larfb_W;

    // assign ptr arrays
    magma_zset_pointer( dR_array,   dR,   lddr,   0, 0, lddr*nb,  batchCount, queue );
    magma_zset_pointer( dT_array,   dT,   lddt,   0, 0, lddt*nb,  batchCount, queue );
    magma_zset_pointer( dW_array,   dW,   lddw,   0, 0, lddw*k,   batchCount, queue );
    magma_zset_pointer( dWvt_array, dWvt, lddwvt, 0, 0, lddwvt*k, batchCount, queue );

    magma_int_t istart, istop, istep;
    if( (side == MagmaLeft  && trans != MagmaNoTrans) ||
        (side == MagmaRight && trans == MagmaNoTrans) ) {
        istart = 0;
        istop  = k;
        istep  = nb;
    }
    else {
        istart = magma_roundup(k-1,nb);
        istop  = 0;
        istep  = -nb;
    }

    // main loop (larft & larfb)
    for(magma_int_t i = istart; (istep < 0 ? i >= istop : i < istop); i+=istep) {
        magma_int_t ib = min(nb, k-i);

        // copy upper tr. part into dR_array
        magmablas_zlacpy_internal_batched(
                MagmaUpper, ib, ib,
                dA_array(i,i), ldda,
                dR_array(i,i), lddr,
                batchCount, queue );

        // set upper tr. part to 1/0
        magmablas_zlaset_internal_batched(
                MagmaUpper, ib, ib,
                MAGMA_Z_ZERO, MAGMA_Z_ONE,
                dA_array(i,i), ldda,
                batchCount, queue );

        // larft
        magma_zlarft_internal_batched(
                Qn-i, ib, 0,
                dA_array(i,i), ldda,
                dtau_array(i),
                dT_array(0,0), lddt,
                dW_array, lddw*k, batchCount, queue);

        // larfb
        // TODO: the check below can be removed if "Quick return" is added to batch larfb
        if(m > i && ib > 0) {
            magma_zlarfb_gemm_internal_batched(
                    side, trans, MagmaForward, MagmaColumnwise,
                    m-i, n, ib,
                    (magmaDoubleComplex_const_ptr*)dA_array(i,i), ldda,
                    (magmaDoubleComplex_const_ptr*)dT_array(0,0), lddt,
                    dC_array(i,0), lddc,
                    dW_array, lddw,
                    dWvt_array, lddwvt,
                    batchCount, queue);
        }

        // copy upper tr. part into dR_array
        magmablas_zlacpy_internal_batched(
                MagmaUpper, ib, ib,
                dR_array(i,i), lddr,
                dA_array(i,i), ldda,
                batchCount, queue );
    }


    return arginfo;

#undef dA_array
#undef dC_array
#undef dR_array
#undef dT_array
#undef dtau_array
} /* magma_zunmqr_batched */
