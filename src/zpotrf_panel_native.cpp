/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

// === Define what BLAS to use ============================================
//    #undef  magma_ztrsm
//    #define magma_ztrsm magmablas_ztrsm
// === End defining what BLAS to use =======================================

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF_RECTILE computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored. (Not currently supported)
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in]
    recnb   INTEGER
            The blocking size at which recursion stops.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[in]
    gbstep  INTEGER
            Internal use.

    @param[out]
    dinfo    INTEGER, stored on the GPU.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @param[out]
    info     INTEGER, stored on the CPU.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    This is an internal routine.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_rectile_native(
    magma_uplo_t uplo, magma_int_t n, magma_int_t recnb,
    magmaDoubleComplex* dA,    magma_int_t ldda, magma_int_t gbstep,
    magma_int_t *dinfo,  magma_int_t *info, magma_queue_t queue)
{
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Constants */
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const double d_one     =  1.0;
    const double d_neg_one = -1.0;

    *info = 0;
    // check arguments
    if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if ( n == 0 ) {
        return *info;
    }

    if (n <= recnb) {
        //  panel factorization
        if(uplo == MagmaLower) {
            magma_zpotf2_lpin(uplo, n, dA, ldda, gbstep, dinfo, queue );
        }
        else {
            magma_zpotf2_native(uplo, n, dA, ldda, gbstep, dinfo, queue );
        }
    }
    else {
        // split A over two [A11 A12;  A21 A22]
        // panel on tile A11,
        // trsm on A21, using A11
        // update on A22 then panel on A22.
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;

        if(uplo == MagmaLower) {
            // panel on A11
            magma_zpotrf_rectile_native(
                uplo, n1, recnb,
                dA(0, 0), ldda,
                gbstep, dinfo, info, queue );

            // TRSM on A21
            magma_ztrsm(
                MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                n2, n1,
                c_one, dA(0, 0), ldda,
                       dA(n1, 0), ldda, queue );

            // update A22
            magma_zherk(
                MagmaLower, MagmaNoTrans, n2, n1,
                d_neg_one, dA(n1, 0), ldda,
                d_one,     dA(n1, n1), ldda, queue );

            // panel on A22
            magma_zpotrf_rectile_native(
                uplo, n2, recnb,
                dA(n1, n1), ldda,
                gbstep+n1, dinfo, info, queue );
        }
        else {
            // uplo = upper
            // panel on A11
            magma_zpotrf_rectile_native(
                uplo, n1, recnb,
                dA(0, 0), ldda,
                gbstep, dinfo, info, queue );

            // TRSM on A12
            magma_ztrsm(
                MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                n1, n2,
                c_one, dA(0,  0), ldda,
                       dA(0, n1), ldda, queue );

            // update A22
            magma_zherk(
                MagmaUpper, MagmaConjTrans, n2, n1,
                d_neg_one, dA(0, n1), ldda,
                d_one,     dA(n1, n1), ldda, queue );

            // panel on A22
            magma_zpotrf_rectile_native(
                uplo, n2, recnb,
                dA(n1, n1), ldda,
                gbstep+n1, dinfo, info, queue );
        }
    }

    return *info;
    #undef dA
}
