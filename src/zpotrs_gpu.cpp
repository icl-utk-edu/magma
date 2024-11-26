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
    ZPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by ZPOTRF.

    This is an expert API, exposing more controls to the user

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by ZPOTRF.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

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

    @param[in]
    queue         magma_queue_t
                  - created/destroyed by the user

    @ingroup magma_potrs
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrs_expert_gpu_work(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_queue_t queue )
{
    // Constants
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    // calculate the required workspace in bytes
    // right now, no workspace is required
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = 0;

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return *info;
    }

    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( nrhs < 0)
        *info = -3;
    else if ( ldda < max(1, n) )
        *info = -5;
    else if ( lddb < max(1, n) )
        *info = -7;
    else if ( *lwork_host < h_workspace_bytes )
        *info = -10;
    else if ( *lwork_device < d_workspace_bytes )
        *info = -12;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return *info;
    }

    if ( uplo == MagmaUpper ) {
        if ( nrhs == 1) {
            magma_ztrsv( MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, dA, ldda, dB, 1, queue );
            magma_ztrsv( MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, dA, ldda, dB, 1, queue );
        } else {
            magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
            magma_ztrsm( MagmaLeft, MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
        }
    }
    else {
        if ( nrhs == 1) {
            magma_ztrsv( MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, dA, ldda, dB, 1, queue );
            magma_ztrsv( MagmaLower, MagmaConjTrans, MagmaNonUnit, n, dA, ldda, dB, 1, queue );
        } else {
            magma_ztrsm( MagmaLeft, MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
            magma_ztrsm( MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
        }
    }

    return *info;
}

/***************************************************************************//**
    Purpose
    -------
    ZPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by ZPOTRF.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by ZPOTRF.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

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

    @ingroup magma_potrs
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info)
{
    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        *info = -1;
    if ( n < 0 )
        *info = -2;
    if ( nrhs < 0)
        *info = -3;
    if ( ldda < max(1, n) )
        *info = -5;
    if ( lddb < max(1, n) )
        *info = -7;
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return *info;
    }

    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // query workspace
    magma_int_t lwork_host[1]   = {-1};
    magma_int_t lwork_device[1] = {-1};
    magma_zpotrs_expert_gpu_work(
        uplo, n, nrhs, NULL, ldda, NULL, lddb,
        info, NULL, lwork_host, NULL, lwork_device,
        queue );

    void* host_work=NULL;
    if( lwork_host[0] > 0 ) {
        magma_malloc_pinned( &host_work, lwork_host[0] );
    }

    void* device_work=NULL;
    if( lwork_device[0] > 0 ) {
        magma_malloc( &device_work, lwork_device[0] );
    }

    magma_zpotrs_expert_gpu_work(
        uplo, n, nrhs, dA, ldda, dB, lddb, info,
        host_work,   lwork_host,
        device_work, lwork_device,
        queue );

    if( host_work != NULL ) magma_free_pinned( host_work );
    if( device_work != NULL ) magma_free( device_work );

    magma_queue_destroy( queue );

    return *info;
}
