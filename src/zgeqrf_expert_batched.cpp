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

    @param[in]
    nb      INTEGER
            The blocking size.

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

    @param[in,out]
    dR_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDR, N/NB)
             dR should be of size (LDDR, N) when provide_RT > 0 and
             of size (LDDT, NB) otherwise. NB is the local blocking size.
             On exit, the elements of R are stored in dR only when provide_RT > 0.

    @param[in]
    lddr     INTEGER
             The leading dimension of the array dR.
             LDDR >= min(M,N) when provide_RT == 1
             otherwise LDDR >= min(NB, min(M,N)).
             NB is the local blocking size.
             To benefit from coalescent memory accesses LDDR must be
             divisible by 16.

    @param[in,out]
    dT_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDT, N/NB)
             dT should be of size (LDDT, N) when provide_RT > 0 and
             of size (LDDT, NB) otherwise. NB is the local blocking size.
             On exit, the elements of T are stored in dT only when provide_RT > 0.

    @param[in]
    lddt     INTEGER
             The leading dimension of the array dT.
             LDDT >= min(NB,min(M,N)). NB is the local blocking size.
             To benefit from coalescent memory accesses LDDR must be
             divisible by 16.

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

    @param[in]
    provide_RT INTEGER
               provide_RT = 0 no R and no T in output.
               dR and dT are used as local workspace to store the R and T of each step.
               provide_RT = 1 the whole R of size (min(M,N), N) and the nbxnb  block of T are provided in output.
               provide_RT = 2 the nbxnb diag block of R and of T are provided in output.

    @param[out]
    dW_array Array of pointers, dimension (2*batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDW, N),
             where LDDW >= NB. Used as a workspace.

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
magma_zgeqrf_expert_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dR_array, magma_int_t lddr,
    magmaDoubleComplex **dT_array, magma_int_t lddt,
    magmaDoubleComplex **dtau_array, magma_int_t provide_RT,
    magmaDoubleComplex **dW_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    #define dA(i, j)  (dA + (i) + (j)*ldda)

    magma_int_t nnb = 8;
    magma_int_t min_mn = min(m, n);
    magma_int_t Tii = 0, Tjj = 0, Rii = 0, Rjj = 0;
    magma_int_t i, ib=nb, jb=nnb, offset_RT=0;
    magma_int_t lddw = nb;

    /* Check arguments */
    magma_int_t arginfo = 0;
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;
    else if (lddr < min_mn && provide_RT == 1)
        arginfo = -6;
    else if (lddr < min(min_mn, nb))
        arginfo = -6;
    else if (lddt < min(min_mn, nb))
        arginfo = -8;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        if (min_mn == 0 ) return arginfo;
    }

    magma_ivec_setc( batchCount, info_array, 0, queue);

    // set dR, dT, and dW to zero. if provide_RT == 0 only a tile of size nbxnb is used and overwritten at each step
    magmablas_zlaset_internal_batched(
            MagmaFull, lddr, (provide_RT > 0 ? n:min(min_mn,nb)), MAGMA_Z_ZERO, MAGMA_Z_ZERO,
            dR_array, 0, 0, lddr, batchCount, queue );
    magmablas_zlaset_internal_batched(
            MagmaFull, lddt, (provide_RT > 0 ? n:min(min_mn,nb)), MAGMA_Z_ZERO, MAGMA_Z_ZERO,
            dT_array, 0, 0, lddt, batchCount, queue );
    magmablas_zlaset_internal_batched(
            MagmaFull, lddw, n, MAGMA_Z_ZERO, MAGMA_Z_ZERO,
            dW_array, 0, 0, lddw, 2*batchCount, queue );

    //magmablas_zlaset( MagmaFull, 2*nb, n*batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dwork, 2*nb, queue );
    for (i=0; i < min_mn; i += nb) {
        ib = min(nb, min_mn-i);

        // panel factorization
        if ( provide_RT > 0 ) {
            offset_RT = i;
            Rii = (provide_RT == 1 ? offset_RT:0);
            Rjj = offset_RT;
            Tii = 0;
            Tjj = offset_RT;
        }

        //dW is used in panel factorization and trailing matrix update
        magma_zgeqrf_panel_internal_batched(m-i, ib, jb,
                                   dA_array,   i, i, ldda,
                                   dtau_array, i,
                                   dT_array, Tii, Tjj, lddt,
                                   dR_array, Rii, Rjj, lddr,
                                   dW_array,
                                   info_array,
                                   batchCount, queue);
        // end of panel

        // update trailing matrix
        if ( (n-ib-i) > 0) {
            // this uses BLAS-3 GEMM routines, different from lapack implementation
            magma_zlarft_internal_batched(
                    m-i, ib, 0,
                    dA_array, i, i, ldda,
                    dtau_array, i,
                    dT_array, 0, (provide_RT > 0) ? offset_RT : 0, lddt,
                    dW_array, nb*lddt,
                    batchCount, queue);

            // perform C = (I-V T^H V^H) * C, C is the trailing matrix
            magma_zlarfb_gemm_internal_batched(
                            MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            m-i, n-i-ib, ib,
                            (const magmaDoubleComplex**)dA_array,   i,   i, ldda,
                            (const magmaDoubleComplex**)dT_array, Tii, Tjj, lddt,
                            dA_array,         i, i+ib, ldda,
                            dW_array, lddw,
                            dW_array + batchCount, lddw,
                            batchCount, queue );
        }
        // end of trailing matrix update

        // copy dR back to V after the trailing matrix update,
        // only when provide_RT=0 otherwise the nbxnb block of V is set to diag=1/0
        // The upper portion of V could be set totaly to 0 here
        if ( provide_RT == 0 ) {
            magmablas_zlacpy_internal_batched(
                    MagmaUpper, ib, ib,
                    dR_array, 0, 0, lddr,
                    dA_array, i, i, ldda,
                    batchCount, queue );
        }
    }

    return arginfo;
}
