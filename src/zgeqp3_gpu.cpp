/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Stan Tomov
       @author Ichitaro Yamazaki
       @author Mark Gates
*/
#include "magma_internal.h"

#define COMPLEX

/***************************************************************************//**
    Purpose
    -------
    ZGEQP3 computes a QR factorization with column pivoting of a
    matrix A:  A*P = Q*R  using Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the upper triangle of the array contains the
            min(M,N)-by-N upper trapezoidal matrix R; the elements below
            the diagonal, together with the array TAU, represent the
            unitary matrix Q as a product of min(M,N) elementary
            reflectors.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted
            to the front of A*P (a leading column); if JPVT(J)=0,
            the J-th column of A is a free column.
            On exit, if JPVT(J)=K, then the J-th column of A*P was the
            the K-th column of A.

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors.

    @param[in,out]
    host_work  Workspace, allocated on host (CPU) memory. For faster CPU-GPU communication,
               user can allocate it as pinned memory using magma_malloc_pinned()

    @param[in,out]
    lwork_host   INTEGER pointer
                 The size of the workspace (host_work) in bytes
                 - lwork_host[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork_host. The workspace itself is not referenced, and no
                   computation is performed.
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
                     computation is performed.
                   - lwork_device[0] >= 0: the routine assumes that the user has provided
                     a workspace with the size in lwork_device.

    @param[out]
    info    INTEGER
      -     = 0: successful exit.
      -     < 0: if INFO = -i, the i-th argument had an illegal value.

    @param[in]
    queue         magma_queue_t
                  - created/destroyed by the user

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector
    with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in
    A(i+1:m,i), and tau in TAU(i).

    @ingroup magma_geqp3
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqp3_expert_gpu_work(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_int_t *info, magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)

    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magma_int_t ione = 1;

    //magma_int_t na;
    magma_int_t n_j;
    magma_int_t j, jb, nb, sm, sn, fjb, nfxd, minmn;
    magma_int_t topbmn, lwkopt;

    minmn = min(m,n);
    nb = magma_get_zgeqp3_nb( m, n );

    // calculate required workspace
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = 0;
    if (minmn == 0) {
        lwkopt = 1;
    } else {
        lwkopt = (n + 1)*nb + 2*n;
    }

    d_workspace_bytes += lwkopt     * sizeof(magmaDoubleComplex);
    d_workspace_bytes += (n+1) * nb * sizeof(magmaDoubleComplex); // df
    d_workspace_bytes += ( 1+256*(n+255)/256 ) * sizeof(double);  // dlsticcs

    // round up to multilpe of sizeof(magmaDoubleComplex)
    // this is potentially important for the top-level interface, which uses lwork
    // as the number of elements, not bytes
    d_workspace_bytes = magma_roundup(d_workspace_bytes, sizeof(magmaDoubleComplex));

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return *info;
    }

    // check arguments
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    else if (*lwork_host < h_workspace_bytes) {
        *info = -8;
    }
    else if (*lwork_device < d_workspace_bytes) {
        *info = -10;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (minmn == 0)
        return *info;

    // assign pointers
    magmaDoubleComplex_ptr dwork=NULL, df=NULL;
    double* dlsticcs=NULL;
    dwork    = (magmaDoubleComplex_ptr)device_work;
    df       = dwork + (n + 1) * nb;
    dlsticcs = (double*)(df    + (n + 1) * nb);
    double *rwork = dlsticcs + (1+256*((n+255)/256));

    magmablas_zlaset( MagmaFull, n+1, nb, c_zero, c_zero, df, n+1, queue );

    nfxd = 0;
    /* Move initial columns up front.
     * Note jpvt uses 1-based indices for historical compatibility. */
    for (j = 0; j < n; ++j) {
        if (jpvt[j] != 0) {
            if (j != nfxd) {
                blasf77_zswap(&m, dA(0, j), &ione, dA(0, nfxd), &ione);  // TODO: ERROR, matrix not on CPU!
                jpvt[j]    = jpvt[nfxd];
                jpvt[nfxd] = j + 1;
            }
            else {
                jpvt[j] = j + 1;
            }
            ++nfxd;
        }
        else {
            jpvt[j] = j + 1;
        }
    }

    /*
        // TODO:
           Factorize fixed columns
           =======================
           Compute the QR factorization of fixed columns and update
           remaining columns.
    if (nfxd > 0) {
        na = min(m,nfxd);
        lapackf77_zgeqrf(&m, &na, dA, &ldda, tau, dwork, &lwork, info);
        if (na < n) {
            n_j = n - na;
            lapackf77_zunmqr( MagmaLeftStr, MagmaConjTransStr, &m, &n_j, &na,
                              dA, &ldda, tau, dA(0, na), &ldda,
                              dwork, &lwork, info );
        }
    }*/

    /*  Factorize free columns */
    if (nfxd < minmn) {
        sm = m - nfxd;
        sn = n - nfxd;
        //sminmn = minmn - nfxd;

        /* Initialize partial column norms. */
        magmablas_dznrm2_cols( sm, sn, dA(nfxd,nfxd), ldda, &rwork[nfxd], queue );
        magma_dcopymatrix( sn, 1, &rwork[nfxd], sn, &rwork[n+nfxd], sn, queue );

        j = nfxd;
        //if (nb < sminmn)
        {
            /* Use blocked code initially. */

            /* Compute factorization: while loop. */
            topbmn = minmn; // - nb;
            while(j < topbmn) {
                jb = min(nb, topbmn - j);

                /* Factorize JB columns among columns J:N. */
                n_j = n - j;

                magma_zlaqps2_gpu   // this is a cuda-file
                    ( m, n_j, j, jb, &fjb,
                      dA(0, j), ldda,
                      &jpvt[j], &tau[j], &rwork[j], &rwork[n + j],
                      dwork,
                      &df[jb], n_j,
                      dlsticcs, queue );

                j += fjb;  /* fjb is actual number of columns factored */
            }
        }

        /*
        // Use unblocked code to factor the last or only block.
        if (j < minmn) {
            n_j = n - j;
            if (j > nfxd) {
                magma_zgetmatrix( m-j, n_j,
                                  dA(j,j), ldda,
                                   A(j,j), lda, queue );
            }
            lapackf77_zlaqp2(&m, &n_j, &j, dA(0, j), &ldda, &jpvt[j],
                             &tau[j], &rwork[j], &rwork[n+j], dwork );
        }*/
    }

    return *info;
} /* magma_zgeqp3_gpu */

/***************************************************************************//**
    Purpose
    -------
    ZGEQP3 computes a QR factorization with column pivoting of a
    matrix A:  A*P = Q*R  using Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the upper triangle of the array contains the
            min(M,N)-by-N upper trapezoidal matrix R; the elements below
            the diagonal, together with the array TAU, represent the
            unitary matrix Q as a product of min(M,N) elementary
            reflectors.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted
            to the front of A*P (a leading column); if JPVT(J)=0,
            the J-th column of A is a free column.
            On exit, if JPVT(J)=K, then the J-th column of A*P was the
            the K-th column of A.

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors.

    @param[out]
    dwork   (workspace) COMPLEX_16 array on the GPU, dimension (MAX(1,LWORK))
            On exit, if INFO=0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            For [sd]geqp3, LWORK >= (N+1)*NB + 2*N;
            for [cz]geqp3, LWORK >= (N+1)*NB,
            where NB is the optimal blocksize.
    \n
            Note: unlike the CPU interface of this routine, the GPU interface
            does not support a workspace query.

*/
#ifdef COMPLEX
/**

    @param
    rwork   (workspace, for [cz]geqp3 only) DOUBLE PRECISION array, dimension (2*N)
            For releases after 2.8.0, this argument is not used, but kept for backward
            compatibility. It can be passed as a null pointer.

*/
#endif // COMPLEX
/**

    @param[out]
    info    INTEGER
      -     = 0: successful exit.
      -     < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector
    with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in
    A(i+1:m,i), and tau in TAU(i).

    @ingroup magma_geqp3
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqp3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dwork, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // query workspace of expert API
    magma_int_t lwork_host[1]   = {-1};
    magma_int_t lwork_device[1] = {-1};
    magma_zgeqp3_expert_gpu_work(
        m, n, NULL, ldda, NULL, NULL,
        NULL, lwork_host, NULL, lwork_device,
        info, queue );

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    else if( lwork*sizeof(magmaDoubleComplex) < lwork_device[0] ) {
        *info = -8;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (min(m,n) == 0)
        return *info;

    magma_zgeqp3_expert_gpu_work(
        m, n, dA, ldda, jpvt, tau,
        NULL, lwork_host, (void*)dwork, lwork_device,
        info, queue );

    magma_queue_sync( queue );
    magma_queue_destroy( queue );

    return *info;
} /* magma_zgeqp3_gpu */
