/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/
#include "magma_internal.h"

//#define MAGMA_PRINTF printf
#define MAGMA_PRINTF(...)
/***************************************************************************//**
    Purpose
    -------
    DSHPOSV computes the solution to a real system of linear equations
        A * X = B,
    where A is an N-by-N symmetric positive definite matrix and X and B
    are N-by-NRHS matrices.

    DSHPOSV first attempts to factorize the matrix in real SINGLE PRECISION, but
    uses a mixed precision matrix multiplication to perform the trailing matrix
    updates (e.g. A_fp16 x B_fp16 ==> C_fp32).
    The routine uses this factorization within an iterative refinement (IR) procedure
    to produce a solution with real DOUBLE PRECISION norm-wise backward error
    quality (see below). The IR procedure has an option to use a GMRES solver to solve
    for the correction vector (Ac = r) instead of a direct solve (r is the residual vector,
    while c is the correction vector).

    Please see for more details:
    ** "Exploiting Lower Precision Arithmetic in Solving Symmetric Positive Definite
        Linear Systems and Least Squares Problems", by Higham et al.
        http://eprints.maths.manchester.ac.uk/2771/

    If the approach fails the method switches to a real DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio real SINGLE PRECISION performance over real DOUBLE PRECISION
    performance is too small. A reasonable strategy should take the
    number of right-hand sides and the size of the matrix into account.
    This might be done with a call to ILAENV in the future. Up to now, we
    always try iterative refinement.

    The iterative refinement process is stopped if
        ITER > ITERMAX
    or for all the RHS we have:
        RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
    where
        o ITER is the number of the current iteration in the iterative
          refinement process
        o RNRM is the infinity-norm of the residual
        o XNRM is the infinity-norm of the solution
        o ANRM is the infinity-operator-norm of the matrix A
        o EPS is the machine epsilon returned by DLAMCH('Epsilon')
    The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00 respectively.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if iterative refinement has been successfully used
            (INFO.EQ.0 and ITER.GE.0, see description below), then A is
            unchanged, if double factorization has been used
            (INFO.EQ.0 and ITER.LT.0, see description below), then the
            array dA contains the factor U or L from the Cholesky
            factorization A = U**T*U or A = L*L**T.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).

    @param[in]
    dB      DOUBLE PRECISION array on the GPU, dimension (LDDB,NRHS)
            The N-by-NRHS right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,N).

    @param[out]
    dX      DOUBLE PRECISION array on the GPU, dimension (LDDX,NRHS)
            If INFO = 0, the N-by-NRHS solution matrix X.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX.  LDDX >= max(1,N).

    @param
    dworkd  (workspace) DOUBLE PRECISION array on the GPU, dimension (N*NRHS)
            This array is used to hold the residual vectors.


    @param
    dworks  (workspace) SINGLE PRECISION array on the GPU, dimension N*(N+nrhs) + N
            This array is used to store the real single precision matrix
            and the right-hand sides or solutions in single precision.

    @param[out]
    iter    INTEGER
      -     < 0: iterative refinement has failed, double precision
                 factorization has been performed
      +     -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
      +     -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
      +     -3 : failure of SPOTRF
      +     -31: stop the iterative refinement after the 30th iteration
      -     > 0: iterative refinement has been successfully used.
                 Returns the number of iterations

    @param[in]
    mode    magma_mode_t
            The mode of the factorization. If mode = MagmaHybrid, then a CPU-GPU
            factorization is used. If mode = MagmaNative, then a GPU-only factorization
            is used.

    @param[in]
    use_gmres  INTEGER
            The solver uses GMRES during iterative refinement if use_gmres > 0.
            Otherwise, classical IR is used.

    @param[in]
    preprocess  INTEGER
            If > 0, the input matrix is scaled/shifted according to:
            http://eprints.maths.manchester.ac.uk/2771/

    @param[in]
    cn      REAL
            A constant that controls the diagonal shift of the matrix (if preprocessing
            is enabled). The diagonal shift is cn * eps_h, where eps_h is the FP16
            unit roundoff.

    @param[in]
    theta   REAL
            A constant that controls the scaling of A to avoid overflow and reduce
            the chances of underflow

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i of (DOUBLE
                  PRECISION) A is not positive definite, so the
                  factorization could not be completed, and the solution
                  has not been computed.

    @ingroup magma_posv
*******************************************************************************/

extern "C" magma_int_t
magma_dshposv_gpu_expert(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_int_t *iter, magma_mode_t mode, magma_int_t use_gmres, magma_int_t preprocess,
    float cn, float theta, magma_int_t *info)
{
    #define dB(i,j)     (dB + (i) + (j)*lddb)
    #define dX(i,j)     (dX + (i) + (j)*lddx)
    #define dR(i,j)     (dR + (i) + (j)*lddr)
    #define dSX(i,j)    (dSX + (i) + (j)*lddsx)

    // Constants
    const double      BWDMAX  = 1.0;
    magma_int_t ITERMAX = 100;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double c_one     = MAGMA_D_ONE;
    const magma_int_t ione  = 1;

    // Local variables
    magmaDouble_ptr dR;
    magmaFloat_ptr dSA, dSX, dD;
    double Xnrmv, Rnrmv;
    double          Anrm, Xnrm, Rnrm, cte, eps, work[1];
    magma_int_t     i, j, iiter, lddsa, lddsx, lddr;

    /* Check arguments */
    *iter = 0;
    *info = 0;
    if ( n < 0 )
        *info = -1;
    else if ( nrhs < 0 )
        *info = -2;
    else if ( ldda < max(1,n))
        *info = -4;
    else if ( lddb < max(1,n))
        *info = -7;
    else if ( lddx < max(1,n))
        *info = -9;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if ( n == 0 || nrhs == 0 )
        return *info;

    lddsa = n;
    lddsx = n;
    lddr  = n;

    dSA = dworks;
    dSX = dSA + lddsa*n;
    dD  = dSX + lddsa * nrhs;
    dR  = dworkd;


    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    eps  = lapackf77_dlamch("Epsilon");
    Anrm = magmablas_dlansy( MagmaInfNorm, uplo, n, dA, ldda, (double*)dworkd, n*nrhs, queue );
    cte  = Anrm * eps * magma_dsqrt( (double) n ) * BWDMAX;


    // for matrix preprocessing, see Higham et al.
    // http://eprints.maths.manchester.ac.uk/2771/
    const float cn_f  = (float)cn;
    const float eps_h = (1.0 / 2048.0); // eps for h
    const float beta  = 1 + (cn_f * eps_h);
    const float miu   = (theta == 0.) ? 1 : ( theta * (float)(65504) ) / beta;

    /*
     * Convert to single precision
     */
    magmablas_dlat2s( uplo, n, dA, ldda, dSA, lddsa, queue, info );
    if (*info != 0) {
        *iter = -2;
        goto fallback;
    }

    if(preprocess > 0) {
        // extract diagonal ans scale
        magmablas_sextract_diag_sqrt(n, n, dSA, lddsa, dD, 1, queue);
        magmablas_sscal_shift_hpd(uplo, n, dSA, lddsa, dD, 1, miu, cn_f, eps_h, queue);
    }

    magmablas_dlag2s( n, nrhs, dB, lddb, dSX, lddsx, queue, info );
    if (*info != 0) {
        *iter = -2;
        goto fallback;
    }

    if(preprocess > 0) {
        magmablas_sdimv_invert(n, MAGMA_S_ONE, dD, 1, dSX, 1, MAGMA_S_ZERO,  dSX, 1, queue);
    }

    // factor dSA in single precision
    if(mode == MagmaHybrid) {
        magma_shpotrf_gpu(uplo, n, dSA, lddsa, info );
    }
    else{
        magma_shpotrf_native(uplo, n, dSA, lddsa, info );
    }

    if (*info != 0) {
        printf("magma_shpotrf_LL_expert_gpu failed, info = %lld \n", (long long)(*info) );
        *iter = -3;
        goto fallback;
    }

    // for cublas symv
    #ifdef MAGMA_HAVE_CUDA
    cublasSetAtomicsMode( queue->cublas_handle(), CUBLAS_ATOMICS_ALLOWED );
    #endif
    if( use_gmres > 0 ) {
        const magma_int_t outer_iter_min = 10, inner_itermax = 50;
        ITERMAX = min(n, outer_iter_min * inner_itermax);
        double gmtol = cte;
        double innertol = 1e-4;
        double rnorm0;
        // FGMRES  GPU need that X is initialized to "0"
        magmablas_dlaset(MagmaFull, n, nrhs, MAGMA_D_ZERO, MAGMA_D_ZERO, dX, lddx, queue);

        magma_dfgmres_spd_gpu(
                uplo, n, nrhs,
                dA, ldda, dSA, lddsa, dD,
                dB, lddb, dX, lddx, dSX,
                ITERMAX, ITERMAX, inner_itermax, inner_itermax,
                gmtol, innertol, &rnorm0, iter, 0, preprocess, miu, queue);
        goto cleanup;
    }
    else {
        // solve dSA*dSX = dB in single precision
        magma_spotrs_gpu( uplo, n, nrhs, dSA, lddsa, dSX, lddsx, info );
        if(preprocess > 0) {
            // scale by miu * D^-1
            magmablas_sdimv_invert(n, miu, dD, 1, dSX, 1, MAGMA_S_ZERO,  dSX, 1, queue);
        }

        // residual dR = dB - dA*dX in double precision
        magmablas_slag2d( n, nrhs, dSX, lddsx, dX, lddx, queue, info );
        magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dR, lddr, queue );
        if ( nrhs == 1 ) {
            magma_dsymv( uplo, n,
                         c_neg_one, dA, ldda,
                                    dX, 1,
                         c_one,     dR, 1, queue );
        }
        else {
            magma_dsymm( MagmaLeft, uplo, n, nrhs,
                         c_neg_one, dA, ldda,
                                    dX, lddx,
                         c_one,     dR, lddr, queue );
        }

        // TODO: use MAGMA_D_ABS( dX(i,j) ) instead of dlange?
        for( j=0; j < nrhs; j++ ) {
            i = magma_idamax( n, dX(0,j), 1, queue ) - 1;
            magma_dgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
            Xnrm = lapackf77_dlange( "F", &ione, &ione, &Xnrmv, &ione, work );

            i = magma_idamax( n, dR(0,j), 1, queue ) - 1;
            magma_dgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );
            Rnrm = lapackf77_dlange( "F", &ione, &ione, &Rnrmv, &ione, work );

            MAGMA_PRINTF("%3d: R = %e\n", 0, Rnrm / (Anrm * n));
            if ( Rnrm >  Xnrm*cte ) {
                goto refinement;
            }
        }

        *iter = 0;
        goto cleanup;
        //return *info;

    refinement:
        for( iiter=1; iiter < ITERMAX; ) {
            *info = 0;
            // convert residual dR to single precision dSX
            magmablas_dlag2s( n, nrhs, dR, lddr, dSX, lddsx, queue, info );
            if (*info != 0) {
                *iter = -2;
                goto fallback;
            }
            // solve dSA*dSX = R in single precision
            if(preprocess > 0) {
                magmablas_sdimv_invert(n, MAGMA_S_ONE, dD, 1, dSX, 1, MAGMA_S_ZERO,  dSX, 1, queue);
            }
            magma_spotrs_gpu( uplo, n, nrhs, dSA, lddsa, dSX, lddsx, info );
            if(preprocess > 0) {
                magmablas_sdimv_invert(n, miu, dD, 1, dSX, 1, MAGMA_S_ZERO,  dSX, 1, queue);
            }

            // Add correction and setup residual
            // dX += dSX [including conversion]  --and--
            // dR = dB
            for( j=0; j < nrhs; j++ ) {
                magmablas_dsaxpycp( n, dSX(0,j), dX(0,j), dB(0,j), dR(0,j), queue );
            }

            // residual dR = dB - dA*dX in double precision
            if ( nrhs == 1 ) {
                magma_dsymv( uplo, n,
                             c_neg_one, dA, ldda,
                                        dX, 1,
                             c_one,     dR, 1, queue );
            }
            else {
                magma_dsymm( MagmaLeft, uplo, n, nrhs,
                             c_neg_one, dA, ldda,
                                        dX, lddx,
                             c_one,     dR, lddr, queue );
            }

            // TODO: use MAGMA_D_ABS( dX(i,j) ) instead of dlange?
            /*  Check whether the nrhs normwise backward errors satisfy the
             *  stopping criterion. If yes, set ITER=IITER > 0 and return. */
            for( j=0; j < nrhs; j++ ) {
                i = magma_idamax( n, dX(0,j), 1, queue ) - 1;
                magma_dgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
                Xnrm = lapackf77_dlange( "F", &ione, &ione, &Xnrmv, &ione, work );

                i = magma_idamax( n, dR(0,j), 1, queue ) - 1;
                magma_dgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );
                Rnrm = lapackf77_dlange( "F", &ione, &ione, &Rnrmv, &ione, work );

                MAGMA_PRINTF("%3d: R = %e\n", iiter, Rnrm / (Anrm * n));
                if ( Rnrm >  Xnrm*cte ) {
                    goto L20;
                }
            }

            /*  If we are here, the nrhs normwise backward errors satisfy
             *  the stopping criterion, we are good to exit. */
            *iter = iiter;
            goto cleanup;
            //return *info;

          L20:
            iiter++;
        }

        /* If we are at this place of the code, this is because we have
         * performed ITER=ITERMAX iterations and never satisified the
         * stopping criterion. Set up the ITER flag accordingly and follow
         * up on double precision routine. */
        *iter = -ITERMAX - 1;

    }
    #ifdef MAGMA_HAVE_CUDA
    cublasSetAtomicsMode( queue->cublas_handle(), CUBLAS_ATOMICS_NOT_ALLOWED );
    #endif

fallback:
    /* Single-precision iterative refinement failed to converge to a
     * satisfactory solution, so we resort to double precision. */
    magma_dpotrf_gpu( uplo, n, dA, ldda, info );
    if (*info == 0) {
        magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dX, lddx, queue );
        magma_dpotrs_gpu( uplo, n, nrhs, dA, ldda, dX, lddx, info );
    }

cleanup:
    magma_queue_destroy( queue );
    return *info;
}


////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dshposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magma_int_t *iter, magma_int_t *info)
{
    float cn    = 0.01;
    float theta = 0.1;

    double *dworkd = NULL;
    float  *dworks = NULL;
    magma_smalloc( &dworks, n*(n+nrhs) + n );  // an extra 'N' is required to store the diagonal
    magma_dmalloc( &dworkd, n*nrhs );

    info = magma_dshposv_gpu_expert(
                uplo, n, nrhs,
                dA, ldda,
                dB, lddb,
                dX, lddx,
                dworkd, dworks,
                iter, MagmaHybrid, 1, 1,
                cn, theta, &info);

    magma_free( dworks );
    magma_free( dworkd );

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dshposv_native(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magma_int_t *iter, magma_int_t *info)
{
    float cn    = 0.01;
    float theta = 0.1;

    double *dworkd = NULL;
    float  *dworks = NULL;
    magma_smalloc( &dworks, n*(n+nrhs) + n );  // an extra 'N' is required to store the diagonal
    magma_dmalloc( &dworkd, n*nrhs );

    info = magma_dshposv_gpu_expert(
                uplo, n, nrhs,
                dA, ldda,
                dB, lddb,
                dX, lddx,
                dworkd, dworks,
                iter, MagmaNative, 1, 1,
                cn, theta, &info);

    magma_free( dworks );
    magma_free( dworkd );

    return 0;
}
