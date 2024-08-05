/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

extern "C" magma_int_t
magma_zgeqrf_batched_work(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dtau_array,
    magma_int_t *info_array,
    void* device_work, magma_int_t* device_lwork,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    /* Local Parameter */
    magma_int_t nb = magma_get_zgeqrf_batched_nb(m);
    magma_int_t min_mn = min(m, n);
    // get a decision on which strategy to use
    magma_int_t use_fused_update = magma_use_zgeqrf_batched_fused_update(m, n, batchCount);

    // aux workspace calculations
    magma_int_t byte_alignment = 128;
    magma_int_t lddt = min(nb, min_mn);
    magma_int_t lddr = min(nb, min_mn);
    magma_int_t sizeR = lddr * lddr * batchCount;
    magma_int_t sizeT = lddt * lddt * batchCount;
    magma_int_t sizeW = 2 * nb * n  * batchCount;

    magma_int_t ptr_count = 0;
    ptr_count += batchCount;     // dR_array
    ptr_count += batchCount;     // dT_array
    ptr_count += batchCount * 2; // dW_array
    ptr_count  = magma_roundup(ptr_count, byte_alignment/sizeof(magmaDoubleComplex*));

    // calculate workspace, if any
    magma_int_t workspace_bytes = 0;
    if(use_fused_update == 0) {
        workspace_bytes += sizeR     * sizeof(magmaDoubleComplex);
        workspace_bytes += sizeT     * sizeof(magmaDoubleComplex);
        workspace_bytes += sizeW     * sizeof(magmaDoubleComplex);
        workspace_bytes += ptr_count * sizeof(magmaDoubleComplex*);
    }

    if(device_lwork[0] < 0) {
        // user is querying workspace
        device_lwork[0] = workspace_bytes;
        return arginfo;
    }

    /* Check arguments */
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;
    else if (device_lwork[0] < workspace_bytes)
        arginfo = -8;
    else if (batchCount < 0)
        arginfo = -9;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || batchCount == 0)
        return arginfo;

    // Try a shortcut that uses fused geqr2 +fused update
    // NOTE: to update the tuning of magma_use_zgeqrf_batched_fused_update
    // (1) disable the for loop below
    // (2) copy the for loop to the testing file (e.g. instead of cublas/hipblas)
    // (3) compare the two solution for different sizes
    if( use_fused_update == 1 ) {
        for(int inb = 16; inb >= 1; inb /= 2) {
            arginfo = magma_zgeqrf_panel_fused_update_batched(
                    m, n, inb,
                    dA_array, 0, 0, ldda,
                    dtau_array, 0, NULL, 0, 0, 0,
                    info_array, 0, batchCount, queue);

            if ( arginfo == 0 ) return arginfo;
        }
    }

    magma_memset(info_array, 0, batchCount*sizeof(magma_int_t));
    magmaDoubleComplex *dT          = NULL;
    magmaDoubleComplex *dR          = NULL;
    magmaDoubleComplex *dW          = NULL;
    magmaDoubleComplex **dW_array   = NULL;
    magmaDoubleComplex **dR_array   = NULL;
    magmaDoubleComplex **dT_array   = NULL;

    dR_array = (magmaDoubleComplex**)device_work;
    dT_array = dR_array + batchCount;
    dW_array = dT_array + batchCount;

    dR = (magmaDoubleComplex*)(dR_array + ptr_count);
    dT = dR + sizeR;
    dW = dT + sizeT;

    magma_zset_pointer( dR_array, dR, lddr, 0, 0, lddr*min(nb, min_mn),   batchCount, queue );
    magma_zset_pointer( dT_array, dT, lddt, 0, 0, lddt*min(nb, min_mn),   batchCount, queue );
    magma_zset_pointer( dW_array, dW,    1, 0, 0, nb*n,                 2*batchCount, queue );

    arginfo = magma_zgeqrf_expert_batched(
                m, n, nb,
                dA_array, ldda,
                dR_array, lddr,
                dT_array, lddt,
                dtau_array, 0,
                dW_array,
                info_array, batchCount, queue);

    return arginfo;
}


/***************************************************************************//**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit, the elements on and above the diagonal of the array
             contain the min(M,N)-by-N upper trapezoidal matrix R (R is
             upper triangular if m >= n); the elements below the diagonal,
             with the array TAU, represent the orthogonal matrix Q as a
             product of min(m,n) elementary reflectors (see Further
             Details).

    @param[in]
    ldda     INTEGER
             The leading dimension of the array dA.  LDDA >= max(1,M).
             To benefit from coalescent memory accesses LDDA must be
             divisible by 16.

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dtau_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t arginfo = 0;

    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return arginfo;

    // query workspace
    magma_int_t device_lwork[1] = {-1};
    void* device_work = NULL;
    magma_zgeqrf_batched_work(m, n, NULL, ldda, NULL, NULL, NULL, device_lwork, batchCount, queue);

    if(device_lwork[0] > 0) {
        magma_malloc((void**)&device_work, device_lwork[0]);
    }

    arginfo = magma_zgeqrf_batched_work( m, n,
                dA_array, ldda, dtau_array,
                info_array, device_work, device_lwork,
                batchCount, queue);

    magma_queue_sync( queue );
    if(device_work != NULL) magma_free(device_work);
    return arginfo;
}
