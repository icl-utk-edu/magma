/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZCGESV computes the solution to a complex system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B,
    where A is an N-by-N matrix and X and B are N-by-NRHS matrices.

    ZCGESV first attempts to factorize the matrix in complex SINGLE PRECISION
    and use this factorization within an iterative refinement procedure 
    to produce a solution with complex DOUBLE PRECISION norm-wise backward error 
    quality (see below). If the approach fails the method switches to a
    complex DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio complex SINGLE PRECISION performance over complex DOUBLE PRECISION
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
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A    * X = B  (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    @param[in]
    n       INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (ldda,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, if iterative refinement has been successfully used
            (info.EQ.0 and ITER.GE.0, see description below), A is
            unchanged. If double precision factorization has been used
            (info.EQ.0 and ITER.LT.0, see description below), then the
            array dA contains the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  ldda >= max(1,N).

    @param[out]
    ipiv    INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).
            Corresponds either to the single precision factorization
            (if info.EQ.0 and ITER.GE.0) or the double precision
            factorization (if info.EQ.0 and ITER.LT.0).

    @param[out]
    dipiv   INTEGER array on the GPU, dimension (N)
            The pivot indices; for 1 <= i <= N, after permuting, row i of the
            matrix was moved to row dIPIV(i).
            Note this is different than IPIV, where interchanges
            are applied one-after-another.

    @param[in]
    dB      COMPLEX_16 array on the GPU, dimension (lddb,NRHS)
            The N-by-NRHS right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  lddb >= max(1,N).

    @param[out]
    dX      COMPLEX_16 array on the GPU, dimension (lddx,NRHS)
            If info = 0, the N-by-NRHS solution matrix X.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX.  lddx >= max(1,N).

    @param
    dworkd  (workspace) COMPLEX_16 array on the GPU, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    @param
    dworks  (workspace) COMPLEX array on the GPU, dimension (N*(N+NRHS))
            This array is used to store the complex single precision matrix
            and the right-hand sides or solutions in single precision.

    @param[out]
    iter    INTEGER
      -     < 0: iterative refinement has failed, double precision
                 factorization has been performed
        +        -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
        +        -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
        +        -3 : failure of SGETRF
        +        -31: stop the iterative refinement after the 30th iteration
      -     > 0: iterative refinement has been successfully used.
                 Returns the number of iterations
 
    @param[out]
    info   INTEGER
      -     = 0:  successful exit
      -     < 0:  if info = -i, the i-th argument had an illegal value
      -     > 0:  if info = i, U(i,i) computed in DOUBLE PRECISION is
                  exactly zero.  The factorization has been completed,
                  but the factor U is exactly singular, so the solution
                  could not be computed.

    @ingroup magma_gesv
*******************************************************************************/
extern "C" magma_int_t
magma_zcgesv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex_ptr dworkd, magmaFloatComplex_ptr dworks,
    magma_int_t *iter,
    magma_int_t *info)
{
    #define dB(i,j)     (dB + (i) + (j)*lddb)
    #define dX(i,j)     (dX + (i) + (j)*lddx)
    #define dR(i,j)     (dR + (i) + (j)*lddr)
    
    // Constants
    const double      BWDMAX  = 1.0;
    const magma_int_t ITERMAX = 30;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magma_int_t ione = 1;
    
    // Local variables
    magmaDoubleComplex_ptr dR;
    magmaFloatComplex_ptr dSA, dSX;
    magmaDoubleComplex Xnrmv, Rnrmv;
    double          Anrm, Xnrm, Rnrm, cte, eps, work[1];
    magma_int_t     i, j, iiter, lddsa, lddr;
    
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
        *info = -8;
    else if ( lddx < max(1,n))
        *info = -10;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    if ( n == 0 || nrhs == 0 )
        return *info;

    lddsa = n;
    lddr  = n;
    
    dSA = dworks;
    dSX = dSA + lddsa*n;
    dR  = dworkd;
    
    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    eps  = lapackf77_dlamch("Epsilon");
    Anrm = magmablas_zlange( MagmaInfNorm, n, n, dA, ldda, (double*)dworkd, n*nrhs, queue );
    cte  = Anrm * eps * magma_dsqrt( (double) n ) * BWDMAX;
    
    /*
     * Convert to single precision
     */
    //magmablas_zlag2c( n, nrhs, dB, lddb, dSX, lddsx, info );  // done inside zcgetrs with pivots
    if (*info != 0) {
        *iter = -2;
        goto fallback;
    }
    
    magmablas_zlag2c( n, n, dA, ldda, dSA, lddsa, queue, info );
    if (*info != 0) {
        *iter = -2;
        goto fallback;
    }
    
    // factor dSA in single precision
    magma_cgetrf_gpu( n, n, dSA, lddsa, ipiv, info );
    if (*info != 0) {
        *iter = -3;
        goto fallback;
    }
    
    // Generate parallel pivots
    {
        magma_int_t *newipiv;
        magma_imalloc_cpu( &newipiv, n );
        if ( newipiv == NULL ) {
            *iter = -3;
            goto fallback;
        }
        magma_swp2pswp( trans, n, ipiv, newipiv );
        magma_isetvector( n, newipiv, 1, dipiv, 1, queue );
        magma_free_cpu( newipiv );
    }
    
    // solve dSA*dSX = dB in single precision
    // converts dB to dSX and applies pivots, solves, then converts result back to dX
    magma_zcgetrs_gpu( trans, n, nrhs, dSA, lddsa, dipiv, dB, lddb, dX, lddx, dSX, info );
    
    // residual dR = dB - dA*dX in double precision
    magmablas_zlacpy( MagmaFull, n, nrhs, dB, lddb, dR, lddr, queue );
    if ( nrhs == 1 ) {
        magma_zgemv( trans, n, n,
                     c_neg_one, dA, ldda,
                                dX, 1,
                     c_one,     dR, 1, queue );
    }
    else {
        magma_zgemm( trans, MagmaNoTrans, n, nrhs, n,
                     c_neg_one, dA, ldda,
                                dX, lddx,
                     c_one,     dR, lddr, queue );
    }
    
    // TODO: use MAGMA_Z_ABS( dX(i,j) ) instead of zlange?
    for( j=0; j < nrhs; j++ ) {
        i = magma_izamax( n, dX(0,j), 1, queue ) - 1;
        magma_zgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
        Xnrm = lapackf77_zlange( "F", &ione, &ione, &Xnrmv, &ione, work );
        
        i = magma_izamax( n, dR(0,j), 1, queue ) - 1;
        magma_zgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );
        Rnrm = lapackf77_zlange( "F", &ione, &ione, &Rnrmv, &ione, work );
        
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
        // solve dSA*dSX = R in single precision
        // convert result back to double precision dR
        // it's okay that dR is used for both dB input and dX output.
        magma_zcgetrs_gpu( trans, n, nrhs, dSA, lddsa, dipiv, dR, lddr, dR, lddr, dSX, info );
        if (*info != 0) {
            *iter = -3;
            goto fallback;
        }
        
        // Add correction and setup residual
        // dX += dR  --and--
        // dR = dB
        // This saves going through dR a second time (if done with one more kernel).
        // -- not really: first time is read, second time is write.
        for( j=0; j < nrhs; j++ ) {
            magmablas_zaxpycp( n, dR(0,j), dX(0,j), dB(0,j), queue );
        }
        
        // residual dR = dB - dA*dX in double precision
        if ( nrhs == 1 ) {
            magma_zgemv( trans, n, n,
                         c_neg_one, dA, ldda,
                                    dX, 1,
                         c_one,     dR, 1, queue );
        }
        else {
            magma_zgemm( trans, MagmaNoTrans, n, nrhs, n,
                         c_neg_one, dA, ldda,
                                    dX, lddx,
                         c_one,     dR, lddr, queue );
        }
        
        // TODO: use MAGMA_Z_ABS( dX(i,j) ) instead of zlange?
        /*  Check whether the nrhs normwise backward errors satisfy the
         *  stopping criterion. If yes, set ITER=IITER > 0 and return. */
        for( j=0; j < nrhs; j++ ) {
            i = magma_izamax( n, dX(0,j), 1, queue ) - 1;
            magma_zgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
            Xnrm = lapackf77_zlange( "F", &ione, &ione, &Xnrmv, &ione, work );
            
            i = magma_izamax( n, dR(0,j), 1, queue ) - 1;
            magma_zgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );
            Rnrm = lapackf77_zlange( "F", &ione, &ione, &Rnrmv, &ione, work );
            
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
    
fallback:
    /* Single-precision iterative refinement failed to converge to a
     * satisfactory solution, so we resort to double precision. */
    magma_zgetrf_gpu( n, n, dA, ldda, ipiv, info );
    if (*info == 0) {
        magmablas_zlacpy( MagmaFull, n, nrhs, dB, lddb, dX, lddx, queue );
        magma_zgetrs_gpu( trans, n, nrhs, dA, ldda, ipiv, dX, lddx, info );
    }
    
cleanup:
    magma_queue_destroy( queue );
    return *info;
}
