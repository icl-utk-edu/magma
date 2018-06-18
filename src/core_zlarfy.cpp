/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
  
       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "magma_bulge.h"

/***************************************************************************//**
 *
 * @ingroup magma_larfy
 *
 *  magma_zlarfy applies an elementary reflector, or Householder matrix, H,
 *  to a n-by-n Hermitian matrix C, from both the left and the right.
 *
 *  H is represented in the form
 *
 *     H = I - tau * v * v'
 *
 *  where  tau  is a scalar and  v  is a vector.
 *
 *  If tau is zero, then H is taken to be the unit matrix.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The number of rows and columns of the matrix C.  n >= 0.
 *
 * @param[in,out] A
 *          COMPLEX*16 array, dimension (lda, n)
 *          On entry, the Hermetian matrix A.
 *          On exit, A is overwritten by H * A * H'.
 *
 * @param[in] lda
 *         The leading dimension of the array A.  lda >= max(1,n).
 *
 * @param[in] V
 *          The vector V that contains the Householder reflectors.
 *
 * @param[in] TAU
 *          The value tau.
 *
 * @param[out] work
 *          Workspace.
 *
 ******************************************************************************/
extern "C" void
magma_zlarfy(
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *V, const magmaDoubleComplex *TAU,
    magmaDoubleComplex *work)
{
    /*
    work (workspace) double complex array, dimension n
    */

    const magma_int_t ione = 1;
    const magmaDoubleComplex c_zero   =  MAGMA_Z_ZERO;
    const magmaDoubleComplex c_neg_one=  MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_half   =  MAGMA_Z_HALF;
    magmaDoubleComplex dtmp;

    /* X = AVtau */
    blasf77_zhemv("L",&n, TAU, A, &lda, V, &ione, &c_zero, work, &ione);

    /* compute dtmp= X'*V */
    dtmp = magma_cblas_zdotc(n, work, ione, V, ione);
    /*
    dtmp = c_zero;
    for (magma_int_t j = 0; j < n; j++)
        dtmp = dtmp + MAGMA_Z_CONJ(work[j]) * V[j];
    */

    /* compute 1/2 X'*V*t = 1/2*dtmp*tau  */
    dtmp = -dtmp * c_half * (*TAU);

    /* compute W=X-1/2VX'Vt = X - dtmp*V */
    blasf77_zaxpy(&n, &dtmp, V, &ione, work, &ione);

    /* performs the symmetric rank 2 operation A := alpha*x*y' + alpha*y*x' + A */
    blasf77_zher2("L", &n, &c_neg_one, work, &ione, V, &ione, A, &lda);
}
