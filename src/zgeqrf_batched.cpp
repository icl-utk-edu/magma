/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "batched_kernel_param.h"


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
    magmaDoubleComplex **dA_array,
    magma_int_t ldda,
    magmaDoubleComplex **dtau_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    #define dA(i, j)  (dA + (i) + (j)*ldda)

    /* Local Parameter */
    magma_int_t nb = magma_get_zgeqrf_batched_nb(m);
    magma_int_t min_mn = min(m, n);

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

    /* Special case for tiny square matrices */
    if( m == n && m <= 32 ){
        return magma_zgeqrf_batched_smallsq( m, dA_array, ldda, dtau_array, info_array, batchCount, queue );
    }

    cudaMemset(info_array, 0, batchCount*sizeof(magma_int_t));

    magmaDoubleComplex *dT        = NULL;
    magmaDoubleComplex *dR        = NULL;
    magmaDoubleComplex **dR_array = NULL;
    magmaDoubleComplex **dT_array = NULL;
    magma_malloc((void**)&dR_array, batchCount * sizeof(*dR_array));
    magma_malloc((void**)&dT_array, batchCount * sizeof(*dT_array));

    magma_int_t lddt = min(nb, min_mn);
    magma_int_t lddr = min(nb, min_mn);
    magma_zmalloc(&dR,  lddr * lddr * batchCount);
    magma_zmalloc(&dT,  lddt * lddt * batchCount);

    /* check allocation */
    if ( dR_array  == NULL || dT_array  == NULL || dR == NULL || dT == NULL ) {
        magma_free(dR_array);
        magma_free(dT_array);
        magma_free(dR);
        magma_free(dT);
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magma_zset_pointer( dR_array, dR, lddr, 0, 0, lddr*min(nb, min_mn), batchCount, queue );
    magma_zset_pointer( dT_array, dT, lddt, 0, 0, lddt*min(nb, min_mn), batchCount, queue );

    arginfo = magma_zgeqrf_expert_batched(m, n,
                                          dA_array, ldda,
                                          dR_array, lddr,
                                          dT_array, lddt,
                                          dtau_array, 0,
                                          info_array, batchCount, queue);

    magma_free(dR_array);
    magma_free(dT_array);
    magma_free(dR);
    magma_free(dT);

    return arginfo;
}
