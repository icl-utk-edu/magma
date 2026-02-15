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

/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the reduced singular value decomposition (SVD) of an M-by-N
    matrix A, optionally computing the left and/or right singular vectors.

    The routine first computes a QR factorization of A, followed by an SVD on the
    R factor. Compared to a direct SVD, better performance is expected on tall-skinny
    matrices.

    The SVD is written as:

         A = U * SIGMA * conjugate-transpose(V)

    where:
    ** SIGMA is a min(m,n)-by-min(m,n) matrix which is zero except for its
    min(m,n) diagonal elements
    ** U is an M-by-min(m,n) unitary matrix, and
    ** V is an N-by-min(m,n) unitary matrix.

    The diagonal elements of SIGMA are the singular values of A; they are real
    and non-negative, and are returned in descending order.
    The first min(m,n) columns of U and V are the left and right singular vectors
    of A.

    NOTES:
    ------
    ** This routines computes only the economy size SVD based on the one-sided
       Jacobi algorithm.

    ** This routines computes V, not V**H (if right vectors are required)

    ** The one-sided Jacobi algorithm implicitly computes the left singular vectors
       anyway while computing the values.

    ** This is the batch version of the routine, which performs the SVD
       on a batch of matrices having the same dimensions.

    Arguments
    ---------
    @param[in]
    jobu    magma_vec_t
            Specifies options for computing all or part of the matrix U:
      -     = MagmaVec or MagmaSomeVec: the first min(m,n) columns of U (the left singular
              vectors) are written.
      -     = MagmaNoVec: no columns of U (no left singular vectors) are
              written to U. However, the algorithm implicitly computes
              them anyway while computing the values.

    @param[in]
    jobv    magma_vec_t
            Specifies options for computing the matrix V:
      -     = MagmaVec or MagmaSomeVec: the first min(m,n) columns of V (the right singular
              vectors) are returned in the array V;
      -     = MagmaNoVec: no columns of V (no right singular vectors) are
              computed.

    @param[in]
    m       INTEGER
            The number of rows of each input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each input matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, length (batchCount).
             Each is a COMPLEX_16 array, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit,
      -      if M >= N and JOBU = MagmaVec or MagmaSomeVec, the user has the option to
             set dU_array = dA_array, upon which A will be overwritten with the
             first min(m,n) columns of U
      -      Otherwise A is unchanged on exit

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,M).

    @param[out]
    dS_array Array of pointers, length (batchCount)
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The singular values of each matrix A, sorted so that S(i) >= S(i+1).

    @param[out]
    dU_array Array of pointers, length (batchCount)
             Each is a COMPLEX_16 array, dimension (LDDU,N)
      -      if JOBU = MagmaVec or MagmaSomeVec, U contains the first min(m,n) columns of U
             (the left singular vectors, stored columnwise);
      -      if JOBU = MagmaNoVec, U is not referenced.
      -      When M >= N, dU_array could optionally be the same as dA_array

    @param[in]
    lddu    INTEGER
            The leading dimension of each array U.  lddu >= max(1,M);

    @param[out]
    dV_array Array of pointers, length (batchCount)
             Each is a COMPLEX_16 array, dimension (LDDV,N)
      -      if JOBV = MagmaVec or MagmaSomeVec, V contains the first min(m,n) columns of V
             (the right singular vectors, stored columnwise);
      -      if JOBV = MagmaNoVec, V is not referenced.

    @param[in]
    lddv    INTEGER
            The leading dimension of each array V.  lddv >= max(1,N);

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.

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

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gesvd
*******************************************************************************/
extern "C" magma_int_t
magma_zgesvj_qr_expert_batched(
    magma_vec_t jobu_org, magma_vec_t jobv_org,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dS_array,
    magmaDoubleComplex** dU_array, magma_int_t lddu,
    magmaDoubleComplex** dV_array, magma_int_t lddv,
    magma_int_t* info_array,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const bool want_us  = (jobu_org == MagmaVec || jobu_org == MagmaSomeVec);
    const bool want_un  = (jobu_org == MagmaNoVec);

    const bool want_vs  = (jobv_org == MagmaVec || jobv_org == MagmaSomeVec);
    const bool want_vn  = (jobv_org == MagmaNoVec);

    // Test the input arguments
    if (! (want_us || want_un) ) {
        arginfo = -1;
    } else if (! (want_vs || want_vn) ) {
        arginfo = -2;
    } else if (morg < 0 ) {
        arginfo = -3;
    } else if (norg < 0) {
        arginfo = -4;
    } else if (ldda < max(1,morg)) {
        arginfo = -6;
    } else if ((lddu < 1) || (want_us && (lddu < morg))) {
        arginfo = -9;
    } else if ((lddv < 1) || (want_vs && (lddv < norg)) ) {
        arginfo = -11;
    } else if (batchCount < 0) {
        arginfo = -15;
    }

    // alignment
    magma_int_t alignment_bytes = 128;
    magma_int_t alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    magma_int_t alignment_int   = alignment_bytes / sizeof(magma_int_t);
    magma_int_t alignment_ptr   = alignment_bytes / sizeof(magmaDoubleComplex*);

    // QR params, if morg < norg, qr is performed on the transpose
    magma_int_t mqr = ( morg >= norg ) ? morg : norg;
    magma_int_t nqr = ( morg >= norg ) ? norg : morg;
    magma_vec_t jobu = ( morg >= norg ) ? jobu_org : jobv_org;
    magma_vec_t jobv = ( morg >= norg ) ? jobv_org : jobu_org;
    magma_int_t minmn_qr = min(mqr, nqr);
    magma_int_t lddqr  = magma_roundup(mqr, 32);

    // qr query workspace
    magma_int_t device_lwork_qr[1] = {-1};
    magma_zgeqrf_batched_work(
        mqr, nqr,
        NULL, lddqr, NULL, NULL, NULL, device_lwork_qr,
        batchCount, queue);

    // svd query workspace, based on nxn matrix size
    int64_t device_lwork_svd[1] = {-1};
    magma_zgesvj_expert_batched(
        jobu, jobv, nqr, nqr, NULL, lddqr, NULL,
        NULL, lddqr, NULL, lddv, NULL, NULL,
        device_lwork_svd, batchCount, queue );

    // unmqr query workspace
    int64_t device_lwork_unmqr[1] = {-1};
    magma_zunmqr_batched(
        MagmaLeft, MagmaNoTrans, mqr, nqr, nqr, NULL, lddqr, NULL,
        NULL, lddqr, NULL, device_lwork_unmqr, NULL, batchCount, queue);

    // calculate workspace of this routine
    size_t ws_qr_v       = magma_roundup( lddqr * nqr * batchCount, alignment );    // copy of A for qr factorization
    size_t ws_qr_r       = magma_roundup( lddqr * nqr * batchCount, alignment );    // to store `R` and pass it to svd
    size_t ws_qr_tau     = magma_roundup( minmn_qr      * batchCount, alignment );    // for tau
    size_t ws_info       = magma_roundup( batchCount, alignment_int);                 // for info in qr and unmqr
    size_t ws_dQR_array  = magma_roundup( batchCount, alignment_ptr);
    size_t ws_dR_array   = magma_roundup( batchCount, alignment_ptr);
    size_t ws_dtau_array = magma_roundup( batchCount, alignment_ptr);

    int64_t device_lwork_svd_qr = 0;
    device_lwork_svd_qr += ws_qr_v       * sizeof(magmaDoubleComplex);
    device_lwork_svd_qr += ws_qr_r       * sizeof(magmaDoubleComplex);
    device_lwork_svd_qr += ws_qr_tau     * sizeof(magmaDoubleComplex);
    device_lwork_svd_qr += ws_info       * sizeof(magma_int_t);
    device_lwork_svd_qr += ws_dQR_array  * sizeof(magmaDoubleComplex*);
    device_lwork_svd_qr += ws_dR_array   * sizeof(magmaDoubleComplex*);
    device_lwork_svd_qr += ws_dtau_array * sizeof(magmaDoubleComplex*);

    int64_t workspace_bytes = device_lwork_svd_qr + max(device_lwork_qr[0], max(device_lwork_svd[0], device_lwork_unmqr[0]));

    // check for workspace query
    if( arginfo == 0 && device_lwork[0] < 0 ) {
        device_lwork[0] = (int64_t)workspace_bytes;
        return arginfo;
    }

    if(arginfo == 0 && device_lwork[0] < workspace_bytes) {
        arginfo = -14;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // split workspace
    void* device_work_svd_qr = device_work;
    void* device_work_other  = (void*)((unsigned char*)device_work_svd_qr +  device_lwork_svd_qr);

    // assign ptrs
    magmaDoubleComplex* dQr   = (magmaDoubleComplex*)device_work_svd_qr;
    magmaDoubleComplex* dR    = dQr + ws_qr_v;
    magmaDoubleComplex* dtau  = dR  + ws_qr_r;
    magma_int_t* dinfo        = (magma_int_t*)(dtau + ws_qr_tau);

    magmaDoubleComplex** dQR_array  = (magmaDoubleComplex**)(dinfo + ws_info);
    magmaDoubleComplex** dR_array   = dQR_array + ws_dQR_array;
    magmaDoubleComplex** dtau_array = dR_array  + ws_dR_array;

    // set workspace to zero
    magma_memset_async(device_work, 0, workspace_bytes, queue);

    // set pointers
    magma_zset_pointer(dQR_array,  dQr,  lddqr,  0, 0, lddqr*nqr,  batchCount, queue );
    magma_zset_pointer(dR_array,   dR,   lddqr,  0, 0, lddqr*nqr,  batchCount, queue );
    magma_zset_pointer(dtau_array, dtau, minmn_qr, 0, 0, minmn_qr, batchCount, queue );

    if(morg >= norg) {
        // copy dA into dQr
        magmablas_zlacpy_batched( MagmaFull, mqr, nqr, dA_array, ldda, dQR_array, lddqr, batchCount, queue );
    }
    else {
        // transpose dA into dQr
        magmablas_ztranspose_conj_batched(
            morg, norg,
            dA_array,  ldda,
            dQR_array, lddqr,
            batchCount, queue );
    }

    // QR factorization
    magma_zgeqrf_batched_work(
        mqr, nqr,
        dQR_array, lddqr, dtau_array,
        dinfo, device_work_other, device_lwork_qr,
        batchCount, queue);

    // copy R factor into dR_array
    magmablas_zlacpy_batched( MagmaUpper, nqr, nqr, dQR_array, lddqr, dR_array, lddqr, batchCount, queue );

    // svd on R
    magmaDoubleComplex** dRightVec_array = (morg >= norg) ? dV_array : dU_array;
    magma_int_t ldd_rightvec             = (morg >= norg) ? lddv : lddu;
    magma_zgesvj_expert_batched(
        jobu, jobv, nqr, nqr, dR_array, lddqr, dS_array,
        dR_array, lddqr, dRightVec_array, ldd_rightvec, info_array,
        device_work_other, device_lwork_svd, batchCount, queue );

    // copy left vectors of `R` then Apply Q
    if(jobu != MagmaNoVec) {
        magmaDoubleComplex** dLeftVec_array = (morg >= norg) ? dU_array : dV_array;
        magma_int_t ldd_leftvec             = (morg >= norg) ? lddu     : lddv;
        magmablas_zlacpy_batched( MagmaFull, mqr, nqr, dR_array, lddqr, dLeftVec_array, ldd_leftvec, batchCount, queue );

        // apply Q on left vectors
        magma_zunmqr_batched(
            MagmaLeft, MagmaNoTrans, mqr, nqr, nqr, dQR_array, lddqr, dtau_array,
            dLeftVec_array, ldd_leftvec, device_work_other, device_lwork_unmqr, dinfo, batchCount, queue);
    }

    return arginfo;
}
