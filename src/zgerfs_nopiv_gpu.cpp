/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define BWDMAX 1.0
#define ITERMAX 30

/***************************************************************************//**
    Purpose
    -------
    ZGERFS improves the computed solution to a system of linear equations.

    The iterative refinement process is stopped if
        ITER > ITERMAX
    or for all the RHS we have:
        RNRM < SQRT(n)*XNRM*ANRM*EPS*BWDMAX
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

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (ldda,N)
            the N-by-N coefficient matrix A.
            
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  ldda >= max(1,N).

    @param[in]
    dB      COMPLEX_16 array on the GPU, dimension (lddb,NRHS)
            The N-by-NRHS right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  lddb >= max(1,N).

    @param[in, out]
    dX      COMPLEX_16 array on the GPU, dimension (lddx,NRHS)
            On entry, the solution matrix X, as computed by
            ZGETRS_NOPIV.  On exit, the improved solution matrix X.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX.  lddx >= max(1,N).

    @param
    dworkd  (workspace) COMPLEX_16 array on the GPU, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    @param
    dAF     COMPLEX*16 array on the GPU, dimension (ldda,n)
            The factors L and U from the factorization A = L*U
            as computed by ZGETRF_NOPIV.

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

    @ingroup magma_gerfs_nopiv
*******************************************************************************/
extern "C" magma_int_t
magma_zgerfs_nopiv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex_ptr dX, magma_int_t lddx,
    magmaDoubleComplex_ptr dworkd, magmaDoubleComplex_ptr dAF,
    magma_int_t *iter,
    magma_int_t *info)
{
    #define dB(i,j)     (dB + (i) + (j)*lddb)
    #define dX(i,j)     (dX + (i) + (j)*lddx)
    #define dR(i,j)     (dR + (i) + (j)*lddr)
    
    /* Constants */
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magma_int_t ione = 1;
    
    /* Local variables */
    magmaDoubleComplex_ptr dR;
    magmaDoubleComplex Xnrmv, Rnrmv;
    double Anrm, Xnrm, Rnrm, cte, eps, work[1];
    magma_int_t i, j, iiter, lddsa, lddr;
    
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

    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    lddsa = n;
    lddr  = n;
    
    dR  = dworkd;
    
    eps  = lapackf77_dlamch("Epsilon");
    Anrm = magmablas_zlange( MagmaInfNorm, n, n, dA, ldda, (magmaDouble_ptr)dworkd, n*nrhs, queue );
    cte  = Anrm * eps * magma_dsqrt( (double) n ) * BWDMAX;
    
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
        //printf("Rnrm : %e, Xnrm*cte : %e\n", Rnrm, Xnrm*cte);
        if ( Rnrm >  Xnrm*cte ) {
            goto refinement;
        }
    }
    
    *iter = 0;
    goto cleanup;

refinement:
    for( iiter=1; iiter < ITERMAX; ) {
        *info = 0;
        // solve dAF*dX = dR
        // it's okay that dR is used for both dB input and dX output.
        magma_zgetrs_nopiv_gpu( trans, n, nrhs, dAF, lddsa, dR, lddr, info );
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
        
      L20:
        iiter++;
    }
    
    /* If we are at this place of the code, this is because we have
     * performed ITER=ITERMAX iterations and never satisified the
     * stopping criterion. Set up the ITER flag accordingly. */
    *iter = -ITERMAX - 1;
    
fallback:
    /* Iterative refinement failed to converge to a
     * satisfactory solution. */
    
cleanup:
    magma_queue_destroy( queue );
    
    return *info;
}
