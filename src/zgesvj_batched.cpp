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

//#define ZGESVJ_TIMER

/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the reduced singular value decomposition (SVD) of an M-by-N
    matrix A , optionally computing the left and/or right singular vectors.
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

    ** This is the internal blocked implementation of the algorithm, which provides
       extra arguments for expert users.

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

    @param[in]
    nb     INTEGER
           The blocking size used by the algorithm. Each input matrix is subdivided into
           block columns of width nb each.

    @param[in]
    max_sweeps    INTEGER
           The maximum number of Jacobi sweeps.

    @param[in]
    heevj_max_sweeps    INTEGER
           The maximum number of Jacobi sweeps for the Hermitian eigensolver used to
           orthogonalize a pair of block columns

    @param[in]
    heevj_tol    DOUBLE
           The tolerance (as multiples of the machine epsilon) for the Hermitian eigensolver.
           This tolerance is used to control if an off-diagonal element in the Gram
           matrix should be annihilated during the Hermitian eigen-decomposition.
           This tolerance can be scaled down by the user as the algorithm progresses
           (see heevj_tol_min, and heevj_tol_scal).

    @param[in]
    heevj_tol_min    DOUBLE
           The minimum tolerance (as multiples of the machine epsilon) for the Hermitian eigensolver.
           The algorithm optionally scales down heevj_tol as long as it is larger than
           heevj_tol_min.

    @param[in]
    heevj_tol_scal    DOUBLE
           A scaling factor for heevj_tol (heevj_tol_scal >= 1).

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
magma_zgesvj_blocked_expert_batched(
    magma_vec_t jobu_org, magma_vec_t jobv_org,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dS_array,
    magmaDoubleComplex** dU_array, magma_int_t lddu,
    magmaDoubleComplex** dV_array, magma_int_t lddv,
    magma_int_t* dinfo_array,
    magma_int_t nb, magma_int_t max_sweeps,
    magma_int_t heevj_max_sweeps, double heevj_tol, double heevj_tol_min, double heevj_tol_scal,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t min_mn  = min(morg, norg);
    const bool want_us  = (jobu_org == MagmaVec || jobu_org == MagmaSomeVec);
    const bool want_un  = (jobu_org == MagmaNoVec);

    const bool want_vs  = (jobv_org == MagmaVec || jobv_org == MagmaSomeVec);
    const bool want_vn  = (jobv_org == MagmaNoVec);

    // the routine supports only morg >= norg, for morg < norg, we do svd on the transpose
    magma_int_t m = ( morg >= norg ) ? morg : norg;
    magma_int_t n = ( morg >= norg ) ? norg : morg;

    // for Jacobi, U is computed anyway, so jobu is always true
    // but if U is not required by the user, we don't copy it from
    // the workspace to dU_array
    magma_vec_t jobv = ( morg >= norg ) ? jobv_org : jobu_org;

    // decide whether to use custom vector updates
    magma_int_t use_custom_vec_updates = (nb <= 32) ? 1 : 0;

    magma_int_t nbx2            = 2 * nb;
    magma_int_t heevj_sort      = 0;  // do not sort eigenvalues or reorder vectors
    magma_int_t heevj_nthreads  = magma_get_zheevj_batched_small_nthreads(nbx2);

    // parameters of blocked implementation
    magma_int_t bpm             = magma_ceildiv(n, nb);  // block columns per matrix
    magma_int_t bpm2            = magma_roundup(bpm, 2); // block columns rounded to be even
    magma_int_t sub_batch       = bpm2 / 2;
    magma_int_t	flat_batchCount = sub_batch * batchCount;
    magma_int_t	n2              = bpm2 * nb;
    magma_int_t iter_per_sweep  = bpm2 - 1;
    int hnot_converged          = 1;   // do not use magma_int_t

    // dimensions
    magma_int_t Um_   = m;
    magma_int_t Un_   = n2;
    magma_int_t Vm_   = n;
    magma_int_t Vn_   = n2;
    magma_int_t Gn_   = nbx2;  // each G is square
    magma_int_t lddu_ = magma_roundup(Um_, 32);
    magma_int_t lddv_ = magma_roundup(Vm_, 32);
    magma_int_t lddg_ = Gn_; // nb is usually a power of 2, so no need to roundup

    // calculate workspace
    magma_int_t alignment_bytes = 128;
    magma_int_t alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    magma_int_t alignment_real  = alignment_bytes / sizeof(double);
    magma_int_t alignment_int   = alignment_bytes / sizeof(magma_int_t);
    magma_int_t alignment_ptr   = alignment_bytes / sizeof(magmaDoubleComplex*);

    size_t vec_copies     = (use_custom_vec_updates == 1) ? 1 : 2;
    size_t ws_eig         = magma_roundup( flat_batchCount * lddg_ * Gn_, alignment );        // eigenvalue problem
    size_t ws_uvec        = magma_roundup( batchCount * lddu_ * Un_, alignment );             // left vectors ws
    size_t ws_vvec        = magma_roundup( batchCount * lddv_ * Vn_, alignment );             // right vectors ws
    size_t ws_info_eig    = magma_roundup( flat_batchCount * iter_per_sweep, alignment_int);  // info array for eigensolver
    size_t ws_sigma       = magma_roundup( batchCount * n,  alignment_real);                  // singular values (unsorted)
    size_t ws_sigma_index = magma_roundup( batchCount * n,  alignment_int);                   // singular values' indices
    size_t ws_eig_sweeps  = magma_roundup( flat_batchCount * iter_per_sweep, alignment_int ); // to record the #sweeps for each heevj problem
    size_t ws_eig_mask    = magma_roundup( flat_batchCount + 1, alignment_int );              // to mask off heevj for already converged problems (+1 for global convergence)
    size_t ws_ptr_array   = magma_roundup( 35 * flat_batchCount, alignment_ptr );             // ptr arrays, no need to align each ptr array individually

    size_t workspace_bytes =  0;
    workspace_bytes += ws_eig         * sizeof(magmaDoubleComplex);
    workspace_bytes += ws_uvec        * sizeof(magmaDoubleComplex) * vec_copies;
    workspace_bytes += ws_vvec        * sizeof(magmaDoubleComplex) * vec_copies;
    workspace_bytes += ws_info_eig    * sizeof(magma_int_t);
    workspace_bytes += ws_sigma       * sizeof(double);
    workspace_bytes += ws_sigma_index * sizeof(magma_int_t);
    workspace_bytes += ws_eig_sweeps  * sizeof(int);    // internal, no need for magma_int_t
    workspace_bytes += ws_eig_mask    * sizeof(int);    // internal, no need for magma_int_t
    workspace_bytes += ws_ptr_array   * sizeof(magmaDoubleComplex*);

    // check for workspace query
    if( device_lwork[0] < 0 ) {
        device_lwork[0] = (int64_t)workspace_bytes;
        return arginfo;
    }

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
    } else if (nb < 2) {
        arginfo = -13;
    } else if (max_sweeps < 1) {
        arginfo = -14;
    } else if (heevj_max_sweeps < 1) {
        arginfo = -15;
    } else if (heevj_tol < 0) {
        arginfo = -16;
    } else if (heevj_tol_min < 0) {
        arginfo = -17;
    } else if (heevj_tol_scal < 1) {
        arginfo = -18;
    } else if(device_lwork[0] < (int64_t)workspace_bytes) {
        arginfo = -20;
    } else if (batchCount < 0) {
        arginfo = -21;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(m == 0 || n == 0 || batchCount == 0) return arginfo;

    #ifdef ZGESVJ_TIMER
    real_Double_t aux_time = 0., gram_time = 0., heevj_time = 0., update_time = 0.;
    real_Double_t time_;
    #endif

    #ifdef ZGESVJ_TIMER
    time_ = magma_sync_wtime( queue );
    #endif

    // assign ptrs in workspace
    magmaDoubleComplex* dG  = (magmaDoubleComplex*)device_work;
    magmaDoubleComplex* dU0 = dG  + ws_eig;
    magmaDoubleComplex* dU1 = (use_custom_vec_updates == 1) ? dU0 : (dU0 + ws_uvec);
    magmaDoubleComplex* dV0 = dU1 + ws_uvec;
    magmaDoubleComplex* dV1 = (use_custom_vec_updates == 1) ? dV0 : (dV0 + ws_vvec);

    // info array for Hermitian eigensolver
    magma_int_t* info_array_eig = (magma_int_t*)(dV1 + ws_vvec);

    // unsorted singular values
    double* dSigma_unsorted = (double*)(info_array_eig + ws_info_eig);
    magma_int_t* dSigma_i   = (magma_int_t*)(dSigma_unsorted + ws_sigma);

    // eig #sweeps and mask
    int* dheevj_nsweeps = (int*)(dSigma_i + ws_sigma_index);
    int* dheevj_mask    = (int*)(dheevj_nsweeps + ws_eig_sweeps);
    int* dnot_converged = (int*)(dheevj_mask + flat_batchCount);   // this single int flag is part of the allocation for dheevj_mask

    // ptr arrays
    magmaDoubleComplex** ptr_array = (magmaDoubleComplex**)(dheevj_mask + ws_eig_mask);

    // for the herk operation (lower only) of each pair [Aj Ak]
    magmaDoubleComplex** dAgemm0_array = ptr_array; // length = (3 x flat_batchCount): [AjxAj AjxAk AkxAk]
    magmaDoubleComplex** dAgemm1_array = dAgemm0_array + 3 * flat_batchCount; // length = (3 x flat_batchCount): [Gjj Gjk Gkk]
    magmaDoubleComplex** dGherk_array  = dAgemm1_array + 3 * flat_batchCount; // length = (3 x flat_batchCount): [Gjj Gjk Gkk]

    // for the eigenvalue problem -- uses Gjj from dGherk_array
    // for eigenvectors, uses the GV00 part of dGV00_GV01_array
    magmaDoubleComplex** dGV00_GV01_array   = dGherk_array     + 3 * flat_batchCount;
    magmaDoubleComplex** dGV10_GV11_array   = dGV00_GV01_array + 4 * flat_batchCount;

    // for updating left/right vectors
    magmaDoubleComplex** dUjVj_input_array    = dGV10_GV11_array   + 4 * flat_batchCount;
    magmaDoubleComplex** dUkVk_input_array    = dUjVj_input_array  + 4 * flat_batchCount;
    magmaDoubleComplex** dUjkVjk_output_array = dUkVk_input_array  + 4 * flat_batchCount;

    // for alternating U & V
    magmaDoubleComplex** dU0_array  = dUjkVjk_output_array + 4 * flat_batchCount;
    magmaDoubleComplex** dU1_array  = dU0_array  + flat_batchCount;
    magmaDoubleComplex** dV0_array  = dU1_array  + flat_batchCount;
    magmaDoubleComplex** dV1_array  = dV0_array  + flat_batchCount;

    // for unsorted singular values
    double** dSigma_unsorted_array   = (double**)(dV1_array + flat_batchCount);
    magma_int_t** dSigma_index_array = (magma_int_t**)(dSigma_unsorted_array + flat_batchCount);

    // init ws to zero
    magma_memset_async(device_work, 0, workspace_bytes, queue);

    // init info to zero
    magma_memset_async(dinfo_array,  0, batchCount * sizeof(magma_int_t), queue);

    // assign ptr arrays that are set once
    // once: output of herk operation (3 x flat_batchCount): [Gjj Gjk Gkk]
    magma_zset_pointer( dGherk_array + (0 * flat_batchCount), dG, lddg_,  0,  0, lddg_*Gn_, flat_batchCount, queue );
    magma_zset_pointer( dGherk_array + (1 * flat_batchCount), dG, lddg_, nb,  0, lddg_*Gn_, flat_batchCount, queue );
    magma_zset_pointer( dGherk_array + (2 * flat_batchCount), dG, lddg_, nb, nb, lddg_*Gn_, flat_batchCount, queue );

    // once: dU0_array, dU1_array, dV0_array, dV1_array, dGjj_array, dGjk_array, dGkk_array
    magma_zset_pointer( dU0_array, dU0, lddu_, 0, 0, lddu_*Un_, batchCount, queue );
    magma_zset_pointer( dU1_array, dU1, lddu_, 0, 0, lddu_*Un_, batchCount, queue );
    magma_zset_pointer( dV0_array, dV0, lddv_, 0, 0, lddv_*Vn_, batchCount, queue );
    magma_zset_pointer( dV1_array, dV1, lddv_, 0, 0, lddv_*Vn_, batchCount, queue );

    // once: dSigma_unsorted_array
    magma_dset_pointer( dSigma_unsorted_array, dSigma_unsorted, n, 0, 0, n, batchCount, queue );
    magma_iset_pointer( dSigma_index_array,    dSigma_i,        n, 0, 0, n, batchCount, queue );

    // once: dGV00_GV01_array:
    // [ GV00, GV01 ]             if V is not required
    // [ GV00, GV01, GV00, GV01 ] if V is required
    magma_zset_pointer( dGV00_GV01_array + 0 * flat_batchCount, dG, lddg_, 0,  0, lddg_*Gn_, flat_batchCount, queue);
    magma_zset_pointer( dGV00_GV01_array + 1 * flat_batchCount, dG, lddg_, 0, nb, lddg_*Gn_, flat_batchCount, queue);
    magma_zset_pointer( dGV00_GV01_array + 2 * flat_batchCount, dG, lddg_, 0,  0, lddg_*Gn_, flat_batchCount, queue);
    magma_zset_pointer( dGV00_GV01_array + 3 * flat_batchCount, dG, lddg_, 0, nb, lddg_*Gn_, flat_batchCount, queue);

    // once: dGV10_GV11_array:
    // [ GV10, GV11 ]             if V is not required
    // [ GV10, GV11, GV10, GV11 ] if V is required
    magma_zset_pointer( dGV10_GV11_array + 0 * flat_batchCount, dG, lddg_, nb,  0, lddg_*Gn_, flat_batchCount, queue);
    magma_zset_pointer( dGV10_GV11_array + 1 * flat_batchCount, dG, lddg_, nb, nb, lddg_*Gn_, flat_batchCount, queue);
    magma_zset_pointer( dGV10_GV11_array + 2 * flat_batchCount, dG, lddg_, nb,  0, lddg_*Gn_, flat_batchCount, queue);
    magma_zset_pointer( dGV10_GV11_array + 3 * flat_batchCount, dG, lddg_, nb, nb, lddg_*Gn_, flat_batchCount, queue);

    // init dUi_array, dUo_array, dVi_array, dVo_array
    magmaDoubleComplex** dUi_array = dU0_array; // i -> input  for GEMM updates
    magmaDoubleComplex** dUo_array = dU1_array; // o -> output for GEMM updates

    magmaDoubleComplex** dVi_array = dV0_array; // i -> input  for GEMM updates
    magmaDoubleComplex** dVo_array = dV1_array; // o -> output for GEMM updates

    // init right vec. to identity
    magmablas_zlaset_batched( MagmaFull, Vm_, Vn_, MAGMA_Z_ZERO, MAGMA_Z_ONE, dVi_array, lddv_, batchCount, queue );

    // copy or transpose dA to dU
    if( morg >= norg ) {
        magmablas_zlacpy_batched( MagmaFull, m, n, dA_array, ldda, dUi_array, lddu_, batchCount, queue );
    }
    else {
        magmablas_ztranspose_conj_batched(
            morg, norg,
            dA_array,  ldda,
            dUi_array, lddu_,
            batchCount, queue );
    }
    magma_memset_async(dnot_converged, sizeof(int), 0, queue);

    #ifdef ZGESVJ_TIMER
    time_     = magma_sync_wtime( queue ) - time_;
    aux_time += time_;
    #endif

    // main loop
    for(magma_int_t isweep = 0; isweep < max_sweeps; isweep++) {
        // loop over iterations per sweep
        for(magma_int_t iter = 0; iter < iter_per_sweep; iter++) {

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue );
            #endif

            // set ptr arrays
            magma_zgesvj_batched_setup_ptr_arrays(
                jobv, iter, nb, bpm2,
                dUi_array, dUo_array, lddu_,
                dVi_array, dVo_array, lddv_,
                dAgemm0_array, dAgemm1_array,
                dUjVj_input_array, dUkVk_input_array, dUjkVjk_output_array, flat_batchCount, queue);

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue ) - time_;
            aux_time += time_;
            #endif

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue );
            #endif
            // three batch gemms consolidated into one call to simulate a batch herk
            magma_zgemm_batched_core(
                MagmaConjTrans, MagmaNoTrans, nb, nb, m,
                MAGMA_Z_ONE,  dAgemm0_array, 0, 0, lddu_,
                              dAgemm1_array, 0, 0, lddu_,
                MAGMA_Z_ZERO, dGherk_array,  0, 0, lddg_,
                3 * flat_batchCount, queue );

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue ) - time_;
            gram_time += time_;
            #endif

            // the eigen-solver mask should be disabled (NULL) in two cases
            // (1) First sweep: since we have no prior information about the orthogonality of any given two block columns
            // (2) If we are using std. GEMM for the vector updates
            int *current_dheevj_mask = (isweep == 0 || use_custom_vec_updates == 0) ? NULL : dheevj_mask;

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue );
            #endif

            // solve eigenvalue problem with vectors, dGherk_array is overwritten with eigenvectors (unless it is masked off)
            magma_zheevj_batched_expert_small_sm(
                MagmaVec, MagmaLower,
                nbx2, dGherk_array, lddg_,
                NULL, info_array_eig + (iter * flat_batchCount),
                current_dheevj_mask, dheevj_nsweeps + (iter * flat_batchCount),
                heevj_tol, heevj_sort, heevj_max_sweeps,
                heevj_nthreads, flat_batchCount, queue );

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue ) - time_;
            heevj_time += time_;
            #endif

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue );
            #endif

            if(use_custom_vec_updates == 1) {
                // Uj & Uk pointers are in the 1st `flat_batchCount` entries in dUjVj_input_array & dUjVj_input_array
                magmaDoubleComplex **dUj_array = dUjVj_input_array;
                magmaDoubleComplex **dUk_array = dUkVk_input_array;

                // update U
                magma_zgesvj_batched_update_vectors(
                    m, nb,
                    dUj_array,    lddu_,
                    dUk_array,    lddu_,
                    dGherk_array, lddg_,
                    info_array_eig + (iter * flat_batchCount),
                    dheevj_nsweeps + (iter * flat_batchCount),
                    flat_batchCount, queue);

                if(jobv == MagmaSomeVec || jobv == MagmaVec) {
                    // Vj & Vk pointers are in the 3rd `flat_batchCount` entries in dUjVj_input_array & dUjVj_input_array
                    magmaDoubleComplex **dVj_array = dUjVj_input_array + (2*flat_batchCount);
                    magmaDoubleComplex **dVk_array = dUkVk_input_array + (2*flat_batchCount);

                    // update V: use 3rd `flat_batchCount` entries from dUjVj_input_array & dUjVj_input_array
                    magma_zgesvj_batched_update_vectors(
                        n, nb,
                        dVj_array,    lddv_,
                        dVk_array,    lddv_,
                        dGherk_array, lddg_,
                        info_array_eig + (iter * flat_batchCount),
                        dheevj_nsweeps + (iter * flat_batchCount),
                        flat_batchCount, queue);
                }
            }
            else {
                // update vectors using std. batch GEMM
                if(m == n) {
                    magma_int_t update_batch = (jobv == MagmaSomeVec || jobv == MagmaVec) ? (4 * flat_batchCount) : (2 * flat_batchCount);
                    // update U & V -- 1st part, four concurrent batch gemms
                    // Ujo = Uj * GV00
                    // Uko = Uj * GV01
                    // Vjo = Vj * GV00 (if jobv != MagmaNoVec)
                    // Vko = Vj * GV01 (if jobv != MagmaNoVec)
                    magma_zgemm_batched_core(
                        MagmaNoTrans, MagmaNoTrans, m, nb, nb,
                        MAGMA_Z_ONE,  dUjVj_input_array,    0, 0, lddu_,
                                      dGV00_GV01_array,     0, 0, lddg_,
                        MAGMA_Z_ZERO, dUjkVjk_output_array, 0, 0, lddu_,
                        update_batch, queue );

                    // update U & V -- 2nd part, four concurrent batch gemms
                    // Ujo += Uk * GV10
                    // Uko += Uk * GV11
                    // Vjo += Vk * GV10 (if jobv != MagmaNoVec)
                    // Vko += Vk * GV11 (if jobv != MagmaNoVec)
                    magma_zgemm_batched_core(
                        MagmaNoTrans, MagmaNoTrans, m, nb, nb,
                        MAGMA_Z_ONE, dUkVk_input_array,    0, 0, lddu_,
                                     dGV10_GV11_array,     0, 0, lddg_,
                        MAGMA_Z_ONE, dUjkVjk_output_array, 0, 0, lddu_,
                        update_batch, queue );

                }
                else {
                    // update U 1st part
                    // Ujo = Uj * GV00
                    // Uko = Uj * GV01
                    magma_zgemm_batched_core(
                        MagmaNoTrans, MagmaNoTrans, m, nb, nb,
                        MAGMA_Z_ONE,  dUjVj_input_array,    0, 0, lddu_,
                                      dGV00_GV01_array,     0, 0, lddg_,
                        MAGMA_Z_ZERO, dUjkVjk_output_array, 0, 0, lddu_,
                        2 * flat_batchCount, queue );

                        // update U 2nd part
                        // Ujo += Uk * GV10
                        // Uko += Uk * GV11
                        magma_zgemm_batched_core(
                            MagmaNoTrans, MagmaNoTrans, m, nb, nb,
                            MAGMA_Z_ONE, dUkVk_input_array,    0, 0, lddu_,
                                         dGV10_GV11_array,     0, 0, lddg_,
                            MAGMA_Z_ONE, dUjkVjk_output_array, 0, 0, lddu_,
                            2 * flat_batchCount, queue );

                    if(jobv == MagmaSomeVec || jobv == MagmaVec) {
                        magma_int_t batch_offset = 2 * flat_batchCount;
                        // update V 1st part
                        // Vjo = Vj * GV00
                        // Vko = Vj * GV01
                        magma_zgemm_batched_core(
                            MagmaNoTrans, MagmaNoTrans, Vm_, nb, nb,
                            MAGMA_Z_ONE,  dUjVj_input_array    + batch_offset, 0, 0, lddv_,
                                          dGV00_GV01_array     + batch_offset, 0, 0, lddg_,
                            MAGMA_Z_ZERO, dUjkVjk_output_array + batch_offset, 0, 0, lddv_,
                            2 * flat_batchCount, queue );

                        // update V 2nd part
                        // Vjo += Vk * GV10
                        // Vko += Vk * GV11
                        magma_zgemm_batched_core(
                            MagmaNoTrans, MagmaNoTrans, Vm_, nb, nb,
                            MAGMA_Z_ONE, dUkVk_input_array    + batch_offset, 0, 0, lddv_,
                                         dGV10_GV11_array     + batch_offset, 0, 0, lddg_,
                            MAGMA_Z_ONE, dUjkVjk_output_array + batch_offset, 0, 0, lddv_,
                            2 * flat_batchCount, queue );
                    }
                }
            }

            #ifdef ZGESVJ_TIMER
            time_     = magma_sync_wtime( queue ) - time_;
            update_time += time_;
            #endif

            // alternate dU and dV for next iteration
            // this has no effect if use_custom_vec_updates = 1,
            // since dU0_array/dV0_array are the same as dU1_array/dV1_array
            dUi_array = (dUi_array == dU0_array) ? dU1_array : dU0_array;
            dUo_array = (dUo_array == dU0_array) ? dU1_array : dU0_array;
            dVi_array = (dVi_array == dV0_array) ? dV1_array : dV0_array;
            dVo_array = (dVo_array == dV0_array) ? dV1_array : dV0_array;
        }

        #ifdef ZGESVJ_TIMER
        time_     = magma_sync_wtime( queue );
        #endif

        // test for convergence
        magma_memset_async(dnot_converged, 0, sizeof(int), queue);
        magma_zgesvj_batched_test_convergence( iter_per_sweep, sub_batch, batchCount, info_array_eig, dheevj_nsweeps, dheevj_mask, dnot_converged, queue );
        magma_getvector(1, sizeof(int), dnot_converged, 1, &hnot_converged, 1, queue);
        if(hnot_converged == 0) break;

        // adjust heevj tol
        heevj_tol = max(heevj_tol / heevj_tol_scal, heevj_tol_min);

        #ifdef ZGESVJ_TIMER
        time_     = magma_sync_wtime( queue ) - time_;
        aux_time += time_;
        #endif
    }

    #ifdef ZGESVJ_TIMER
    time_     = magma_sync_wtime( queue );
    #endif

    // finalize values (batch column-wise norms)
    // should use dUi_array
    magma_zgesvj_batched_finalize_values(m, n, dUi_array, lddu_, dSigma_unsorted_array, batchCount, queue);

    // sort the values into dS_array
    magmablas_dsort_batched(
        MagmaDescending, min_mn,
        dSigma_unsorted_array, 1,
        dS_array,              1,
        dSigma_index_array, batchCount, queue);

    // finalize vectors
    magma_zgesvj_batched_finalize_vectors(
        jobu_org, jobv_org, morg, norg,
        dUi_array, lddu_, dVi_array, lddv_,
        dU_array,  lddu,  dV_array,  lddv,
        dS_array, dSigma_index_array, batchCount, queue);

    #ifdef ZGESVJ_TIMER
    time_     = magma_sync_wtime( queue ) - time_;
    aux_time += time_;
    #endif

    #ifdef ZGESVJ_TIMER
    real_Double_t total_time = aux_time + gram_time + heevj_time + update_time;
    printf("stats %5d   %5d   %5d   %8.4f   %8.4f   %8.4f   %8.4f   %8.4f\n",
          batchCount, morg, norg, aux_time*1000., gram_time*1000.,
          heevj_time*1000., update_time*1000., total_time*1000.);
    #endif

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the reduced singular value decomposition (SVD) of an M-by-N
    matrix A , optionally computing the left and/or right singular vectors.
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
magma_zgesvj_expert_batched(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dS_array,
    magmaDoubleComplex** dU_array, magma_int_t lddu,
    magmaDoubleComplex** dV_array, magma_int_t lddv,
    magma_int_t* dinfo_array,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const bool want_us  = (jobu == MagmaVec || jobu == MagmaSomeVec);
    const bool want_un  = (jobu == MagmaNoVec);

    const bool want_vs  = (jobv == MagmaVec || jobv == MagmaSomeVec);
    const bool want_vn  = (jobv == MagmaNoVec);

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

    // params for blocked code
    // supports only morg >= norg. for morg < norg, we do svd on the transpose
    magma_int_t m = ( morg >= norg ) ? morg : norg;
    magma_int_t n = ( morg >= norg ) ? norg : morg;
    magma_int_t gesvj_nb         = magma_get_zgesvj_batched_nb( m, n );
    magma_int_t gesvj_max_sweeps = 100;
    double eps                   = lapackf77_dlamch("E");
    double heevj_tol             = pow(10,floor(0.3 * log10(1/eps)));  // starting tolerance (as multiples of eps) for heevj, empirically decided
    double heevj_tol_min         = 30; // minimum tolerance (as multiples of eps) for heevj
    double heevj_tol_scal        = 10; // heevj_tol is divided by heevj_tol_scal after every Jacobi sweep (to disable, set heevj_tol to desired tolerance and set heevj_tol_scal to 1)
    magma_int_t heevj_max_sweeps = 1;  // partial or full eigensolver (a full solver impacts performance)


    // calculate workspace, if any
    bool use_fused_gesvj = magma_zgesvj_batched_use_fused( jobu, jobv, morg, norg );
    int64_t workspace_bytes[1] = {-1};
    if( use_fused_gesvj ) {
        workspace_bytes[0] = 0;
    } else {
        // query workspace of blocked code
        magma_zgesvj_blocked_expert_batched(
            jobu, jobv, morg, norg,
            NULL, ldda, NULL, NULL, lddu, NULL, lddv, NULL,
            gesvj_nb, gesvj_max_sweeps,
            heevj_max_sweeps, heevj_tol, heevj_tol_min, heevj_tol_scal,
            NULL, workspace_bytes, batchCount, queue);
    }

    // check for workspace query
    if(device_lwork[0] < 0) {
        device_lwork[0] = workspace_bytes[0];
        arginfo = 0;
        return arginfo;
    }

    // execute batch svd
    // first, check workspace size
    if( device_lwork[0] < workspace_bytes[0] ) {
        arginfo = -14;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(morg == 0 || norg == 0 || batchCount == 0) return arginfo;

    if( use_fused_gesvj ) {
        // fused code
        arginfo = magma_zgesvj_batched_small_sm(
                    jobu, jobv, morg, norg,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv,
                    dinfo_array, batchCount, queue );
    } else {
        // blocked code
        arginfo = magma_zgesvj_blocked_expert_batched(
                    jobu, jobv, morg, norg,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv, dinfo_array,
                    gesvj_nb, gesvj_max_sweeps,
                    heevj_max_sweeps, heevj_tol, heevj_tol_min, heevj_tol_scal,
                    device_work, device_lwork,
                    batchCount, queue );
    }

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the reduced singular value decomposition (SVD) of an M-by-N
    matrix A , optionally computing the left and/or right singular vectors.
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

    @param[in]
    strideA    INTEGER
            The stride (in elements) between two consecutive A matrices.
            strideA >= (LDDA*N).

    @param[out]
    dS       Pointer to the beginning of an array of pointers whose length is (batchCount),
             such that S[i+1] = S[i] + strideS
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The singular values of each matrix A, sorted so that S(i) >= S(i+1).

    @param[in]
    strideS    INTEGER
            The stride (in elements) between two consecutive S vectors.
            strideS >= MIN(M, N).

    @param[out]
    dU       Pointer to the beginning of an array of pointers whose length is (batchCount),
             such that U[i+1] = U[i] + strideU
             Each is a COMPLEX_16 array, dimension (LDDU,N)
      -      if JOBU = MagmaVec or MagmaSomeVec, U contains the first min(m,n) columns of U
             (the left singular vectors, stored columnwise);
      -      if JOBU = MagmaNoVec, U is not referenced.
      -      When M >= N, dU_array could optionally be the same as dA_array

    @param[in]
    lddu    INTEGER
            The leading dimension of each array U.  lddu >= max(1,M);

    @param[in]
    strideU    INTEGER
            The stride (in elements) between two consecutive U matrices.
            strideU >= (LDDU * MIN(M,N)).

    @param[out]
    dV       Pointer to the beginning of an array of pointers whose length is (batchCount),
             such that V[i+1] = V[i] + strideV
             Each is a COMPLEX_16 array, dimension (LDDV,N)
      -      if JOBV = MagmaVec or MagmaSomeVec, V contains the first n columns of V
             (the right singular vectors, stored columnwise);
      -      if JOBV = MagmaNoVec, V is not referenced.

    @param[in]
    lddv    INTEGER
            The leading dimension of each array V.  lddv >= max(1,N);

    @param[in]
    strideV    INTEGER
            The stride (in elements) between two consecutive V matrices.
            strideU >= (LDDV * MIN(M,N)).

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
extern "C"
magma_int_t
magma_zgesvj_expert_batched_strided(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t strideA,
    magmaDouble_ptr        dS, magma_int_t strideS,
    magmaDoubleComplex_ptr dU, magma_int_t lddu, magma_int_t strideU,
    magmaDoubleComplex_ptr dV, magma_int_t lddv, magma_int_t strideV,
    magmaInt_ptr dinfo_array,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t min_mn  = min(morg, norg);
    const bool want_us  = (jobu == MagmaVec || jobu == MagmaSomeVec);
    const bool want_un  = (jobu == MagmaNoVec);

    const bool want_vs  = (jobv == MagmaVec || jobv == MagmaSomeVec);
    const bool want_vn  = (jobv == MagmaNoVec);

    magma_int_t min_strideA = ldda * norg;
    magma_int_t min_strideS = min_mn;
    magma_int_t min_strideU = lddu * min_mn;
    magma_int_t min_strideV = lddv * min_mn;

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
    } else if ( strideA < min_strideA ) {
        arginfo = -7;
    } else if ( strideS < min_strideS ) {
        arginfo = -9;
    } else if ((lddu < 1) || (want_us && (lddu < morg))) {
        arginfo = -11;
    } else if ( strideU < min_strideU ) {
        arginfo = -12;
    } else if ((lddv < 1) || (want_vs && (lddv < norg)) ) {
        arginfo = -14;
    } else if ( strideV < min_strideV ) {
        arginfo = -15;
    } else if (batchCount < 0) {
        arginfo = -19;
    }

    // params for blocked code
    // supports only morg >= norg. for morg < norg, we do svd on the transpose
    magma_int_t m = ( morg >= norg ) ? morg : norg;
    magma_int_t n = ( morg >= norg ) ? norg : morg;
    magma_int_t gesvj_nb         = magma_get_zgesvj_batched_nb( m, n );
    magma_int_t gesvj_max_sweeps = 100;
    double eps                   = lapackf77_dlamch("E");
    double heevj_tol             = pow(10,floor(0.3 * log10(1/eps)));  // starting tolerance (as multiples of eps) for heevj, empirically decided
    double heevj_tol_min         = 30; // minimum tolerance (as multiples of eps) for heevj
    double heevj_tol_scal        = 10; // heevj_tol is divided by heevj_tol_scal after every Jacobi sweep (to disable, set heevj_tol to desired tolerance and set heevj_tol_scal to 1)
    magma_int_t heevj_max_sweeps = 1;  // partial or full eigensolver (a full solver impacts performance)


    // calculate workspace, if any
    bool use_fused_gesvj = magma_zgesvj_batched_use_fused( jobu, jobv, morg, norg );
    int64_t workspace_bytes[1] = {-1};
    if( use_fused_gesvj ) {
        workspace_bytes[0] = 0;
    } else {
        // query workspace of blocked code
        magma_zgesvj_blocked_expert_batched(
            jobu, jobv, morg, norg,
            NULL, ldda, NULL, NULL, lddu, NULL, lddv, NULL,
            gesvj_nb, gesvj_max_sweeps,
            heevj_max_sweeps, heevj_tol, heevj_tol_min, heevj_tol_scal,
            NULL, workspace_bytes, batchCount, queue);
    }

    // for stride interface, we need workspace to generate the pointer arrays
    // for dA_array, dS_array, dU_array, and dV_array
    int64_t ws_ptr_array = magma_roundup( 4 * batchCount * sizeof(magmaDoubleComplex*), 128);
    workspace_bytes[0] += batchCount * ws_ptr_array;

    // check for workspace query
    if(device_lwork[0] < 0) {
        device_lwork[0] = workspace_bytes[0];
        arginfo = 0;
        return arginfo;
    }

    // execute batch svd
    // first, check workspace size
    if( device_lwork[0] < workspace_bytes[0] ) {
        arginfo = -18;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(morg == 0 || norg == 0 || batchCount == 0) return arginfo;

    // generate ptr arrays
    magmaDoubleComplex ** dA_array = (magmaDoubleComplex**)(device_work);
    magmaDoubleComplex ** dU_array = dA_array + batchCount;
    magmaDoubleComplex ** dV_array = dU_array + batchCount;
    double**              dS_array = (double**)(dV_array + batchCount);
    magma_zset_pointer( dA_array, dA, ldda, 0, 0, strideA, batchCount, queue );
    magma_zset_pointer( dU_array, dU, lddu, 0, 0, strideU, batchCount, queue );
    magma_zset_pointer( dV_array, dV, lddv, 0, 0, strideV, batchCount, queue );
    magma_dset_pointer( dS_array, dS,    1, 0, 0, strideS, batchCount, queue );

    void* device_work_    = (void**)(dA_array  + ws_ptr_array/sizeof(magmaDoubleComplex*));
    int64_t device_lwork_ = device_lwork[0] - ws_ptr_array;

    if( use_fused_gesvj ) {
        // fused code
        arginfo = magma_zgesvj_batched_small_sm(
                    jobu, jobv, morg, norg,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv,
                    dinfo_array, batchCount, queue );
    } else {
        // blocked code
        arginfo = magma_zgesvj_blocked_expert_batched(
                    jobu, jobv, morg, norg,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv, dinfo_array,
                    gesvj_nb, gesvj_max_sweeps,
                    heevj_max_sweeps, heevj_tol, heevj_tol_min, heevj_tol_scal,
                    device_work_, &device_lwork_,
                    batchCount, queue );
    }

    return arginfo;
}


/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the reduced singular value decomposition (SVD) of an M-by-N
    matrix A , optionally computing the left and/or right singular vectors.
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

    ** This is the internal blocked implementation of the algorithm, which provides
       extra arguments for expert users.

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

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gesvd
*******************************************************************************/
extern "C" magma_int_t
magma_zgesvj_batched(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dS_array,
    magmaDoubleComplex** dU_array, magma_int_t lddu,
    magmaDoubleComplex** dV_array, magma_int_t lddv,
    magma_int_t* dinfo_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const bool want_us  = (jobu == MagmaVec || jobu == MagmaSomeVec);
    const bool want_un  = (jobu == MagmaNoVec);

    const bool want_vs  = (jobv == MagmaVec || jobv == MagmaSomeVec);
    const bool want_vn  = (jobv == MagmaNoVec);

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
        arginfo = -13;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(morg == 0 || norg == 0 || batchCount == 0) return arginfo;


    // query qworkspace
    void* device_work = NULL;
    int64_t device_lwork[1] = {-1};
    magma_zgesvj_expert_batched(
        jobu, jobv, morg, norg,
        NULL, ldda, NULL, NULL, lddu, NULL, lddv, dinfo_array,
        device_work, device_lwork,
        batchCount, queue );

    if(device_lwork[0] > 0) {
        magma_malloc(&device_work, device_lwork[0]);
    }

    // main call
    magma_zgesvj_expert_batched(
        jobu, jobv, morg, norg,
        dA_array, ldda, dS_array,
        dU_array, lddu,
        dV_array, lddv, dinfo_array,
        device_work, device_lwork,
        batchCount, queue );

    if(device_work != NULL) magma_free( device_work );
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the reduced singular value decomposition (SVD) of an M-by-N
    matrix A , optionally computing the left and/or right singular vectors.
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

    ** This is the internal blocked implementation of the algorithm, which provides
       extra arguments for expert users.

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

    @param[in]
    strideA    INTEGER
            The stride (in elements) between two consecutive A matrices.
            strideA >= (LDDA*N).

    @param[out]
    dS       Pointer to the beginning of an array of pointers whose length is (batchCount),
             such that S[i+1] = S[i] + strideS
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The singular values of each matrix A, sorted so that S(i) >= S(i+1).

    @param[in]
    strideS    INTEGER
            The stride (in elements) between two consecutive S vectors.
            strideS >= MIN(M, N).

    @param[out]
    dU       Pointer to the beginning of an array of pointers whose length is (batchCount),
             such that U[i+1] = U[i] + strideU
             Each is a COMPLEX_16 array, dimension (LDDU,N)
      -      if JOBU = MagmaVec or MagmaSomeVec, U contains the first min(m,n) columns of U
             (the left singular vectors, stored columnwise);
      -      if JOBU = MagmaNoVec, U is not referenced.
      -      When M >= N, dU_array could optionally be the same as dA_array

    @param[in]
    lddu    INTEGER
            The leading dimension of each array U.  lddu >= max(1,M);

    @param[in]
    strideU    INTEGER
            The stride (in elements) between two consecutive U matrices.
            strideU >= (LDDU * MIN(M,N)).

    @param[out]
    dV       Pointer to the beginning of an array of pointers whose length is (batchCount),
             such that V[i+1] = V[i] + strideV
             Each is a COMPLEX_16 array, dimension (LDDV,N)
      -      if JOBV = MagmaVec or MagmaSomeVec, V contains the first n columns of V
             (the right singular vectors, stored columnwise);
      -      if JOBV = MagmaNoVec, V is not referenced.

    @param[in]
    lddv    INTEGER
            The leading dimension of each array V.  lddv >= max(1,N);

    @param[in]
    strideV    INTEGER
            The stride (in elements) between two consecutive V matrices.
            strideU >= (LDDV * MIN(M,N)).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gesvd
*******************************************************************************/
extern "C"
magma_int_t
magma_zgesvj_batched_strided(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t strideA,
    magmaDouble_ptr        dS, magma_int_t strideS,
    magmaDoubleComplex_ptr dU, magma_int_t lddu, magma_int_t strideU,
    magmaDoubleComplex_ptr dV, magma_int_t lddv, magma_int_t strideV,
    magmaInt_ptr dinfo_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t min_mn  = min(morg, norg);
    const bool want_us  = (jobu == MagmaVec || jobu == MagmaSomeVec);
    const bool want_un  = (jobu == MagmaNoVec);

    const bool want_vs  = (jobv == MagmaVec || jobv == MagmaSomeVec);
    const bool want_vn  = (jobv == MagmaNoVec);

    magma_int_t min_strideA = morg * norg;
    magma_int_t min_strideS = min_mn;
    magma_int_t min_strideU = morg * min_mn;
    magma_int_t min_strideV = min_mn * norg;

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
    } else if ( strideA < min_strideA ) {
        arginfo = -7;
    } else if ( strideS < min_strideS ) {
        arginfo = -9;
    } else if ((lddu < 1) || (want_us && (lddu < morg))) {
        arginfo = -11;
    } else if ( strideU < min_strideU ) {
        arginfo = -12;
    } else if ((lddv < 1) || (want_vs && (lddv < norg)) ) {
        arginfo = -14;
    } else if ( strideV < min_strideV ) {
        arginfo = -15;
    } else if (batchCount < 0) {
        arginfo = -17;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if(morg == 0 || norg == 0 || batchCount == 0) return arginfo;


    // query qworkspace
    void* device_work = NULL;
    int64_t device_lwork[1] = {-1};
    magma_zgesvj_expert_batched_strided(
        jobu, jobv, morg, norg,
        NULL, ldda, strideA, NULL, strideS, NULL, lddu, strideU, NULL, lddv, strideV, NULL,
        device_work, device_lwork,
        batchCount, queue );

    if(device_lwork[0] > 0) {
        magma_malloc(&device_work, device_lwork[0]);
    }

    // main call
    magma_zgesvj_expert_batched_strided(
        jobu, jobv, morg, norg,
        dA, ldda, strideA, dS, strideS,
        dU, lddu, strideU,
        dV, lddv, strideV, dinfo_array,
        device_work, device_lwork,
        batchCount, queue );

    if(device_work != NULL) magma_free( device_work );
    return arginfo;
}
