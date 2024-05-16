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
    ZGETRS solves a system of linear equations
        A * X = B,
        A**T * X = B,  or
        A**H * X = B
    with a general N-by-N matrix A using the LU factorization computed by ZGETRF_GPU.

    This is an expert interface, which exposes more controls to the user.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A    * X = B  (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The factors L and U from the factorization A = P*L*U as computed
            by ZGETRF_GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    ipiv    INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1 <= i <= N, row i of the
            matrix was interchanged with row IPIV(i).

    @param[in,out]
    dB      COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @param[in]
    mode    magma_mode_t
      -     = MagmaNative: Factorize dA using GPU only mode (currently not implemented);
      -     = MagmaHybrid: Factorize dA using Hybrid (CPU/GPU) mode.

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
    queue         magma_queue_t
                  - created/destroyed by the user outside the routine
                  - Used for kernel execution

    @ingroup magma_getrs
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrs_expert_gpu_work(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info,
    magma_mode_t mode,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_queue_t queue )
{
    // Constants
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    // Local variables
    magmaDoubleComplex *work = NULL;
    bool notran = (trans == MagmaNoTrans);
    magma_int_t i1, i2, inc;

    // calculate the required workspace in bytes
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = 0;
    if (mode == MagmaHybrid) {
        h_workspace_bytes += n * nrhs * sizeof(magmaDoubleComplex);
    }
    else {
        // native mode, not currently supported
        d_workspace_bytes += 0;
    }

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return 0;
    }

    *info = 0;
    if ( (! notran) &&
         (trans != MagmaTrans) &&
         (trans != MagmaConjTrans) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -8;
    } else if ( mode != MagmaHybrid ) {
        printf("ERROR: function %s only supported hybrid mode\n", __func__);
        *info = -10;
    }
    else if ( *lwork_host < h_workspace_bytes ) {
        *info = -12;
    }
    else if ( *lwork_device < d_workspace_bytes ) {
        *info = -14;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    // Assign pointers
    work = (magmaDoubleComplex*)host_work;

    i1 = 1;
    i2 = n;
    if (notran) {
        inc = 1;

        /* Solve A * X = B. */
        magma_zgetmatrix( n, nrhs, dB, lddb, work, n, queue );
        lapackf77_zlaswp( &nrhs, work, &n, &i1, &i2, ipiv, &inc );
        magma_zsetmatrix( n, nrhs, work, n, dB, lddb, queue );

        if ( nrhs == 1) {
            magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaUnit,    n, dA, ldda, dB, 1, queue );
            magma_ztrsv( MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, dA, ldda, dB, 1, queue );
        } else {
            magma_ztrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,    n, nrhs, c_one, dA, ldda, dB, lddb, queue );
            magma_ztrsm( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
        }
    } else {
        inc = -1;

        /* Solve A**T * X = B  or  A**H * X = B. */
        if ( nrhs == 1) {
            magma_ztrsv( MagmaUpper, trans, MagmaNonUnit, n, dA, ldda, dB, 1, queue );
            magma_ztrsv( MagmaLower, trans, MagmaUnit,    n, dA, ldda, dB, 1, queue );
        } else {
            magma_ztrsm( MagmaLeft, MagmaUpper, trans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
            magma_ztrsm( MagmaLeft, MagmaLower, trans, MagmaUnit,    n, nrhs, c_one, dA, ldda, dB, lddb, queue );
        }

        magma_zgetmatrix( n, nrhs, dB, lddb, work, n, queue );
        lapackf77_zlaswp( &nrhs, work, &n, &i1, &i2, ipiv, &inc );
        magma_zsetmatrix( n, nrhs, work, n, dB, lddb, queue );
    }

    return *info;
}

/***************************************************************************//**
    Purpose
    -------
    ZGETRS solves a system of linear equations
        A * X = B,
        A**T * X = B,  or
        A**H * X = B
    with a general N-by-N matrix A using the LU factorization computed by ZGETRF_GPU.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A    * X = B  (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The factors L and U from the factorization A = P*L*U as computed
            by ZGETRF_GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    ipiv    INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1 <= i <= N, row i of the
            matrix was interchanged with row IPIV(i).

    @param[in,out]
    dB      COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_getrs
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info)
{

    // Local variables
    void *host_work = NULL, *device_work=NULL;
    bool notran = (trans == MagmaNoTrans);

    magma_mode_t mode = MagmaHybrid;

    *info = 0;
    if ( (! notran) &&
         (trans != MagmaTrans) &&
         (trans != MagmaConjTrans) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -8;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // query workspace
    magma_int_t lwork_host[1]   = {-1};
    magma_int_t lwork_device[1] = {-1};
    magma_zgetrs_expert_gpu_work(
         trans, n, nrhs, NULL, ldda, NULL, NULL, lddb, info,
         mode, NULL, lwork_host, NULL, lwork_device, queue );

    if(lwork_host[0] > 0) {
        magma_malloc_cpu( &host_work, lwork_host[0] );
    }

    if(lwork_device[0] > 0) {
        magma_malloc( &device_work, lwork_device[0] );
    }

    magma_zgetrs_expert_gpu_work(
        trans, n, nrhs,
        dA, ldda, ipiv,
        dB, lddb,
        info,
        mode,
        host_work,   lwork_host,
        device_work, lwork_device,
        queue );
    magma_queue_sync( queue );

    magma_queue_destroy( queue );

    if(! (host_work   == NULL) ) magma_free_cpu( host_work );
    if(! (device_work == NULL) ) magma_free( device_work );

    return *info;
}
