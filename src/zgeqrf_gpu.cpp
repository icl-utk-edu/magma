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
    Auxiliary function: "A" is pointer to the current panel holding the
    Householder vectors for the QR factorization of the panel. This routine
    puts ones on the diagonal and zeros in the upper triangular part of "A".
    The upper triangular values are stored in work.

    Then, the inverse is calculated in place in work, so as a final result,
    work holds the inverse of the upper triangular diagonal block.
*******************************************************************************/
void zsplit_diag_block_invert(
    magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *work )
{
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    magma_int_t i, j, info;
    magmaDoubleComplex *cola, *colw;

    for (i=0; i < ib; i++) {
        cola = A    + i*lda;
        colw = work + i*ib;
        for (j=0; j < i; j++) {
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
    lapackf77_ztrtri( MagmaUpperStr, MagmaNonUnitStr, &ib, work, &ib, &info );
}

/***************************************************************************//**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted and
    stored separately in dT.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
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
    dT      (workspace) COMPLEX_16 array on the GPU,
            dimension (2*MIN(M, N) + ceil(N/32)*32 )*NB,
            where NB can be obtained through magma_get_zgeqrf_nb( M, N ).
            It starts with a MIN(M,N)*NB block that stores the triangular T
            matrices, followed by a MIN(M,N)*NB block that stores inverses of
            the diagonal blocks of the R matrix.
            The rest of the array is used as workspace.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @param[in]
    mode    magma_mode_t
      -     Only mode = MagmaHybrid is currently supported, factorizes dA using Hybrid (CPU/GPU) mode.

    @param[in]
    nb      INTEGER
            The blocking size used during the factorization. nb > 0;
            Users with no specific preference of nb can call magma_get_zgeqrf_nb()
            to get the value of nb as determined by MAGMA's internal tuning.

    @param[in,out]
    host_work  Workspace, allocated on host (CPU) memory. For faster CPU-GPU communication,
               user can allocate it as pinned memory using magma_malloc_pinned()

    @param[in,out]
    lwork_host   INTEGER pointer
                 The size of the workspace (host_work) in bytes
                 - lwork_host[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork_host. The workspace itself is not referenced, and no
                   factorization is performed.
                -  lwork[0] >= 0: the routine assumes that the user has provided
                   a workspace with the size in lwork_host.

    @param[in,out]
    device_work  Workspace, allocated on device (GPU) memory.

    @param[in,out]
    lwork_device   INTEGER pointer
                   The size of the workspace (device_work) in bytes
                   - lwork_device[0] < 0: a workspace query is assumed, the routine
                     calculates the required amount of workspace and returns
                     it in lwork_device. The workspace itself is not referenced, and no
                     factorization is performed.
                   - lwork_device[0] >= 0: the routine assumes that the user has provided
                     a workspace with the size in lwork_device.

    @param[in]
    queues        magma_queue_t array of size two
                  - created/destroyed by the user outside the routine
                  - Used for concurrent kernel execution, if possible


    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v^H

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_expert_gpu_work(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT,
    magma_int_t *info,
    magma_mode_t mode, magma_int_t nb,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_queue_t queues[2] )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
    #define dT(i_)     (dT + (i_)*nb)
    #define dR(i_)     (dT + (  minmn + (i_))*nb)
    #define dwork(i_)  (dT + (2*minmn + (i_))*nb)

    magmaDoubleComplex *work, *hwork, *R;
    magma_int_t cols, i, ib, ldwork, lddwork, lhwork, lwork, minmn, old_i, old_ib, rows;

    minmn = min( m, n );

    // ---- compute lddwork, ldwork, lhwork, and lwork
    // dT contains 3 blocks:
    // dT    is minmn*nb
    // dR    is minmn*nb
    // dwork is n*nb
    lddwork = n;

    // work  is m*nb for panel
    // hwork is n*nb, and at least nb*nb for T in larft
    // R     is nb*nb
    ldwork = m;
    lhwork = max( n*nb, nb*nb );
    lwork  = ldwork*nb + lhwork + nb*nb;
    // last block needs rows*cols for matrix and prefers cols*nb for work
    // worst case is n > m*nb, m a small multiple of nb:
    // needs n*nb + n > (m+n)*nb
    // prefers 2*n*nb, about twice above (m+n)*nb.
    i = ((minmn-1)/nb)*nb;
    lwork = max( lwork, (m-i)*(n-i) + (n-i)*nb );

    // calculate the required workspace in bytes
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = 0;
    if (nb <= 1 || nb >= minmn) {
        h_workspace_bytes += 2*m*n*sizeof(magmaDoubleComplex);
    }
    else {
        h_workspace_bytes += lwork * sizeof(magmaDoubleComplex);
    }

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return *info;
    }

    // check arguments
    *info = 0;
    if        ( m < 0 ) {
        *info = -1;
    }
    else if ( n < 0 ) {
        *info = -2;
    }
    else if ( ldda < max(1,m) ) {
        *info = -4;
    }
    else if ( mode != MagmaHybrid ) {
        printf("%s is only available in hybrid mode", __func__);
        *info = -8;
    }
    else if (nb < 1) {
        *info = -9;
    }
    else if( *lwork_host   < h_workspace_bytes ) {
        *info = -11;
    }
    else if( *lwork_device < d_workspace_bytes ) {
        *info = -13;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (minmn == 0)
        return *info;

    if (nb <= 1 || nb >= minmn) {
        /* Use CPU code. */
        work = (magmaDoubleComplex*)host_work;
        magma_zgetmatrix(m, n, dA, ldda, work, m, queues[0] );
        lhwork = m*n;
        lapackf77_zgeqrf( &m, &n, work, &m, tau, work+m*n, &lhwork, info );
        magma_zsetmatrix( m, n, work, m, dA, ldda, queues[0] );
        return *info;
    }

    // use blocked code
    work = (magmaDoubleComplex*)host_work;
    hwork = work + ldwork*nb;
    R     = work + ldwork*nb + lhwork;
    memset( R, 0, nb*nb*sizeof(magmaDoubleComplex) );

    if ( nb > 1 && nb < minmn ) {
        // need nb*nb for T in larft
        assert( lhwork >= nb*nb );

        // Use blocked code initially
        old_i = 0; old_ib = nb;
        for (i = 0; i < minmn-nb; i += nb) {
            ib = min( minmn-i, nb );
            rows = m - i;

            // get i-th panel from device
            magma_zgetmatrix_async( rows, ib,
                                    dA(i,i), ldda,
                                    work,    ldwork, queues[1] );
            if (i > 0) {
                // Apply H^H to A(i:m,i+2*ib:n) from the left
                cols = n - old_i - 2*old_ib;
                magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, cols, old_ib,
                                  dA(old_i, old_i         ), ldda, dT(old_i), nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0),  lddwork, queues[0] );

                // Fix the diagonal block
                magma_zsetmatrix_async( old_ib, old_ib,
                                        R,         old_ib,
                                        dR(old_i), old_ib, queues[0] );
            }

            magma_queue_sync( queues[1] );  // wait to get work(i)
            lapackf77_zgeqrf( &rows, &ib, work, &ldwork, &tau[i], hwork, &lhwork, info );
            // Form the triangular factor of the block reflector in hwork
            // H = H(i) H(i+1) . . . H(i+ib-1)
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work, &ldwork, &tau[i], hwork, &ib );

            // wait for previous trailing matrix update (above) to finish with R
            magma_queue_sync( queues[0] );

            // copy the upper triangle of panel to R and invert it, and
            // set  the upper triangle of panel (V) to identity
            zsplit_diag_block_invert( ib, work, ldwork, R );

            // send i-th V matrix to device
            magma_zsetmatrix( rows, ib,
                              work, ldwork,
                              dA(i,i), ldda, queues[1] );

            if (i + ib < n) {
                // send T matrix to device
                magma_zsetmatrix( ib, ib,
                                  hwork, ib,
                                  dT(i), nb, queues[1] );

                if (i+nb < minmn-nb) {
                    // Apply H^H to A(i:m,i+ib:i+2*ib) from the left
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    // wait for larfb to finish with dwork before larfb in next iteration starts
                    magma_queue_sync( queues[1] );
                }
                else {
                    // Apply H^H to A(i:m,i+ib:n) from the left
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    // Fix the diagonal block
                    magma_zsetmatrix( ib, ib,
                                      R,     ib,
                                      dR(i), ib, queues[1] );
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    // Use unblocked code to factor the last or only block.
    if (i < minmn) {
        rows = m-i;
        cols = n-i;
        magma_zgetmatrix( rows, cols, dA(i, i), ldda, work, rows, queues[1] );
        // see comments for lwork above
        lhwork = lwork - rows*cols;
        lapackf77_zgeqrf( &rows, &cols, work, &rows, &tau[i], &work[rows*cols], &lhwork, info );
        magma_zsetmatrix( rows, cols, work, rows, dA(i, i), ldda, queues[1] );
    }

    return *info;
} // magma_zgeqrf_gpu_exper_work


/***************************************************************************//**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted and
    stored separately in dT.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
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
    dT      (workspace) COMPLEX_16 array on the GPU,
            dimension (2*MIN(M, N) + ceil(N/32)*32 )*NB,
            where NB can be obtained through magma_get_zgeqrf_nb( M, N ).
            It starts with a MIN(M,N)*NB block that stores the triangular T
            matrices, followed by a MIN(M,N)*NB block that stores inverses of
            the diagonal blocks of the R matrix.
            The rest of the array is used as workspace.

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

        H(i) = I - tau * v * v^H

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info )
{
    // check arguments
    *info = 0;
    if        ( m < 0 ) {
        *info = -1;
    }
    else if ( n < 0 ) {
        *info = -2;
    }
    else if ( ldda < max(1,m) ) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_int_t minmn = min(m, n);
    if (minmn == 0)
        return *info;

    // queues
    magma_device_t cdev;
    magma_queue_t queues[2];
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );


    magma_mode_t mode = MagmaHybrid;
    magma_int_t  nb   = magma_get_zgeqrf_nb(m, n);

    // query workspace
    void *host_work = NULL, *device_work=NULL;
    magma_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
    magma_zgeqrf_expert_gpu_work(
        m, n, NULL, ldda,
        NULL, NULL, info,
        mode, nb,
        NULL, lhwork,
        NULL, ldwork, queues );
    // alloc workspace
    if( lhwork[0] > 0 ) {
        magma_malloc_pinned( (void**)&host_work, lhwork[0] );
    }

    if( ldwork[0] > 0 ) {
        magma_malloc( (void**)&device_work, ldwork[0] );
    }

    magma_zgeqrf_expert_gpu_work(
        m, n, dA, ldda, tau, dT, info,
        mode, nb,
        host_work, lhwork, device_work, ldwork, queues );
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );

    // free workspace
    if( host_work != NULL) {
        magma_free_pinned( host_work );
    }

    if( device_work != NULL ) {
        magma_free( device_work );
    }

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    return *info;
} // magma_zgeqrf_gpu
