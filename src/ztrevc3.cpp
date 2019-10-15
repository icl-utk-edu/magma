/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @author Azzam Haidar
       
       @precisions normal z -> c
*/
#include "magma_timer.h"

#include "magma_internal.h"

#define COMPLEX

/***************************************************************************//**
    Purpose
    -------
    ZTREVC3 computes some or all of the right and/or left eigenvectors of
    a complex upper triangular matrix T.
    Matrices of this type are produced by the Schur factorization of
    a complex general matrix:  A = Q*T*Q**H, as computed by ZHSEQR.

    The right eigenvector x and the left eigenvector y of T corresponding
    to an eigenvalue w are defined by:

                 T*x = w*x,     (y**H)*T = w*(y**H)

    where y**H denotes the conjugate transpose of the vector y.
    The eigenvalues are not input to this routine, but are read directly
    from the diagonal of T.

    This routine returns the matrices X and/or Y of right and left
    eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
    input matrix. If Q is the unitary factor that reduces a matrix
    A to Schur form T, then Q*X and Q*Y are the matrices of right and
    left eigenvectors of A.

    This uses a Level 3 BLAS version of the back transformation.

    Arguments
    ---------
    @param[in]
    side     magma_side_t
       -     = MagmaRight:      compute right eigenvectors only;
       -     = MagmaLeft:       compute left eigenvectors only;
       -     = MagmaBothSides:  compute both right and left eigenvectors.

    @param[in]
    howmany  magma_vec_t
       -     = MagmaAllVec:        compute all right and/or left eigenvectors;
       -     = MagmaBacktransVec:  compute all right and/or left eigenvectors,
                                   backtransformed by the matrices in VR and/or VL;
       -     = MagmaSomeVec:       compute selected right and/or left eigenvectors,
                                   as indicated by the logical array select.

    @param[in]
    select   LOGICAL array, dimension (n)
             If howmany = MagmaSomeVec, select specifies the eigenvectors to be
             computed.
             The eigenvector corresponding to the j-th eigenvalue is
             computed if select[j] = true.
             Not referenced if howmany = MagmaAllVec or MagmaBacktransVec.

    @param[in]
    n        INTEGER
             The order of the matrix T. n >= 0.

    @param[in,out]
    T        COMPLEX_16 array, dimension (ldt,n)
             The upper triangular matrix T. modified, but restored on exit.

    @param[in]
    ldt      INTEGER
             The leading dimension of the array T. ldt >= max(1,n).

    @param[in,out]
    VL       COMPLEX_16 array, dimension (ldvl,mm)
             On entry, if side = MagmaLeft or MagmaBothSides and howmany = MagmaBacktransVec, VL must
             contain an n-by-n matrix Q (usually the unitary matrix Q
             of Schur vectors returned by ZHSEQR).
             On exit, if side = MagmaLeft or MagmaBothSides, VL contains:
             if howmany = MagmaAllVec, the matrix Y of left eigenvectors of T;
             if howmany = MagmaBacktransVec, the matrix Q*Y;
             if howmany = MagmaSomeVec, the left eigenvectors of T specified by
                              select, stored consecutively in the columns
                              of VL, in the same order as their eigenvalues.
             Not referenced if side = MagmaRight.

    @param[in]
    ldvl     INTEGER
             The leading dimension of the array VL.
             ldvl >= 1, and if side = MagmaLeft or MagmaBothSides, ldvl >= n.

    @param[in,out]
    VR       COMPLEX_16 array, dimension (ldvr,mm)
             On entry, if side = MagmaRight or MagmaBothSides and howmany = MagmaBacktransVec, VR must
             contain an n-by-n matrix Q (usually the unitary matrix Q
             of Schur vectors returned by ZHSEQR).
             On exit, if side = MagmaRight or MagmaBothSides, VR contains:
             if howmany = MagmaAllVec, the matrix X of right eigenvectors of T;
             if howmany = MagmaBacktransVec, the matrix Q*X;
             if howmany = MagmaSomeVec, the right eigenvectors of T specified by
                               select, stored consecutively in the columns
                               of VR, in the same order as their eigenvalues.
             Not referenced if side = MagmaLeft.

    @param[in]
    ldvr     INTEGER
             The leading dimension of the array VR.
             ldvr >= 1, and if side = MagmaRight or MagmaBothSides, ldvr >= n.

    @param[in]
    mm       INTEGER
             The number of columns in the arrays VL and/or VR. mm >= mout.

    @param[out]
    mout     INTEGER
             The number of columns in the arrays VL and/or VR actually
             used to store the eigenvectors.
             If howmany = MagmaAllVec or MagmaBacktransVec, mout is set to n.
             Each selected eigenvector occupies one column.

    @param[out]
    work     COMPLEX_16 array, dimension (max(1,lwork))

    @param[in]
    lwork    INTEGER
             The dimension of array work. lwork >= max(1,2*n).
             For optimum performance, lwork >= (1 + 2*nb)*n, where nb is
             the optimal blocksize.

    @param[out]
    rwork    double array, dimension (n)

    @param[out]
    info     INTEGER
       -     = 0:  successful exit
       -     < 0:  if info = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The algorithm used in this program is basically backward (forward)
    substitution, with scaling to make the the code robust against
    possible overflow.
    
    Each eigenvector is normalized so that the element of largest
    magnitude has magnitude 1; here the magnitude of a complex number
    (x,y) is taken to be |x| + |y|.

    @ingroup magma_trevc
*******************************************************************************/
extern "C"
magma_int_t magma_ztrevc3(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select,  // logical in Fortran
    magma_int_t n,
    magmaDoubleComplex *T,  magma_int_t ldt,
    magmaDoubleComplex *VL, magma_int_t ldvl,
    magmaDoubleComplex *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{
    #define  T(i,j)  ( T + (i) + (j)*ldt )
    #define VL(i,j)  (VL + (i) + (j)*ldvl)
    #define VR(i,j)  (VR + (i) + (j)*ldvr)
    #define work(i,j) (work + (i) + (j)*n)

    // .. Parameters ..
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    const magma_int_t  nbmin = 16, nbmax = 128;
    const magma_int_t  ione = 1;
    
    // .. Local Scalars ..
    magma_int_t            allv, bothv, leftv, over, rightv, somev;
    magma_int_t            i, ii, is, j, k, ki, iv, n2, nb, nb2, version;
    double                 ovfl, remax, scale, smin, smlnum, ulp, unfl;
    
    // Decode and test the input parameters
    bothv  = (side == MagmaBothSides);
    rightv = (side == MagmaRight) || bothv;
    leftv  = (side == MagmaLeft ) || bothv;

    allv  = (howmany == MagmaAllVec);
    over  = (howmany == MagmaBacktransVec);
    somev = (howmany == MagmaSomeVec);

    // Set mout to the number of columns required to store the selected
    // eigenvectors.
    if ( somev ) {
        *mout = 0;
        for( j=0; j < n; ++j ) {
            if ( select[j] ) {
                *mout += 1;
            }
        }
    }
    else {
        *mout = n;
    }

    *info = 0;
    if ( ! rightv && ! leftv )
        *info = -1;
    else if ( ! allv && ! over && ! somev )
        *info = -2;
    else if ( n < 0 )
        *info = -4;
    else if ( ldt < max( 1, n ) )
        *info = -6;
    else if ( ldvl < 1 || ( leftv && ldvl < n ) )
        *info = -8;
    else if ( ldvr < 1 || ( rightv && ldvr < n ) )
        *info = -10;
    else if ( mm < *mout )
        *info = -11;
    else if ( lwork < max( 1, 2*n ) )
        *info = -14;
    
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible.
    if ( n == 0 ) {
        return *info;
    }
    
    // Use blocked version (2) if sufficient workspace.
    // Requires 1 vector to save diagonal elements, and 2*nb vectors for x and Q*x.
    // (Compared to dtrevc3, rwork stores 1-norms.)
    // Zero-out the workspace to avoid potential NaN propagation.
    nb = 2;
    if ( lwork >= n + 2*n*nbmin ) {
        version = 2;
        nb = (lwork - n) / (2*n);
        nb = min( nb, nbmax );
        nb2 = 1 + 2*nb;
        lapackf77_zlaset( "F", &n, &nb2, &c_zero, &c_zero, work, &n );
    }
    else {
        version = 1;
    }

    // Set the constants to control overflow.
    unfl = lapackf77_dlamch( "Safe minimum" );
    ovfl = 1. / unfl;
    lapackf77_dlabad( &unfl, &ovfl );
    ulp = lapackf77_dlamch( "Precision" );
    smlnum = unfl*( n / ulp );

    // Store the diagonal elements of T in working array work.
    for( i=0; i < n; ++i ) {
        *work(i,0) = *T(i,i);
    }

    // Compute 1-norm of each column of strictly upper triangular
    // part of T to control overflow in triangular solver.
    rwork[0] = 0.;
    for( j=1; j < n; ++j ) {
        rwork[j] = magma_cblas_dzasum( j, T(0,j), ione );
    }

    magma_timer_t time_total=0, time_trsv=0, time_gemm=0, time_gemv=0, time_trsv_sum=0, time_gemm_sum=0, time_gemv_sum=0;
    timer_start( time_total );

    if ( rightv ) {
        // ============================================================
        // Compute right eigenvectors.
        // iv is index of column in current block.
        // Non-blocked version always uses iv=1;
        // blocked     version starts with iv=nb, goes down to 1.
        // (Note the "0-th" column is used to store the original diagonal.)
        iv = 1;
        if ( version == 2 ) {
            iv = nb;
        }
        
        timer_start( time_trsv );
        is = *mout - 1;
        for( ki=n-1; ki >= 0; --ki ) {
            if ( somev ) {
                if ( ! select[ki] ) {
                    continue;
                }
            }
            smin = max( ulp*( MAGMA_Z_ABS1( *T(ki,ki) ) ), smlnum );

            // --------------------------------------------------------
            // Complex right eigenvector
            *work(ki,iv) = c_one;

            // Form right-hand side.
            for( k=0; k < ki; ++k ) {
                *work(k,iv) = -(*T(k,ki));
            }

            // Solve upper triangular system:
            // [ T(1:ki-1,1:ki-1) - T(ki,ki) ]*X = scale*work.
            for( k=0; k < ki; ++k ) {
                *T(k,k) = *T(k, k) - *T(ki,ki);
                if ( MAGMA_Z_ABS1( *T(k,k) ) < smin ) {
                    *T(k,k) = MAGMA_Z_MAKE( smin, 0. );
                }
            }

            if ( ki > 0 ) {
                lapackf77_zlatrs( "Upper", "No transpose", "Non-unit", "Y",
                                  &ki, T, &ldt,
                                  work(0,iv), &scale, rwork, info );
                *work(ki,iv) = MAGMA_Z_MAKE( scale, 0. );
            }

            // Copy the vector x or Q*x to VR and normalize.
            if ( ! over ) {
                // ------------------------------
                // no back-transform: copy x to VR and normalize
                n2 = ki+1;
                blasf77_zcopy( &n2, work(0,iv), &ione, VR(0,is), &ione );

                ii = blasf77_izamax( &n2, VR(0,is), &ione ) - 1;
                remax = 1. / MAGMA_Z_ABS1( *VR(ii,is) );
                blasf77_zdscal( &n2, &remax, VR(0,is), &ione );

                for( k=ki+1; k < n; ++k ) {
                    *VR(k,is) = c_zero;
                }
            }
            else if ( version == 1 ) {
                // ------------------------------
                // version 1: back-transform each vector with GEMV, Q*x.
                time_trsv_sum += timer_stop( time_trsv );
                timer_start( time_gemv );
                if ( ki > 0 ) {
                    blasf77_zgemv( "n", &n, &ki, &c_one,
                                   VR, &ldvr,
                                   work(0, iv), &ione,
                                   work(ki,iv), VR(0,ki), &ione );
                }
                time_gemv_sum += timer_stop( time_gemv );
                ii = blasf77_izamax( &n, VR(0,ki), &ione ) - 1;
                remax = 1. / MAGMA_Z_ABS1( *VR(ii,ki) );
                blasf77_zdscal( &n, &remax, VR(0,ki), &ione );
                timer_start( time_trsv );
            }
            else if ( version == 2 ) {
                // ------------------------------
                // version 2: back-transform block of vectors with GEMM
                // zero out below vector
                for( k=ki+1; k < n; ++k ) {
                    *work(k,iv) = c_zero;
                }

                // Columns iv:nb of work are valid vectors.
                // When the number of vectors stored reaches nb,
                // or if this was last vector, do the GEMM
                if ( (iv == 1) || (ki == 0) ) {
                    time_trsv_sum += timer_stop( time_trsv );
                    timer_start( time_gemm );
                    nb2 = nb-iv+1;
                    n2  = ki+nb-iv+1;
                    blasf77_zgemm( "n", "n", &n, &nb2, &n2, &c_one,
                                   VR, &ldvr,
                                   work(0,iv   ), &n, &c_zero,
                                   work(0,nb+iv), &n );
                    time_gemm_sum += timer_stop( time_gemm );
                    
                    // normalize vectors
                    // TODO if somev, should copy vectors individually to correct location.
                    for( k = iv; k <= nb; ++k ) {
                        ii = blasf77_izamax( &n, work(0,nb+k), &ione ) - 1;
                        remax = 1. / MAGMA_Z_ABS1( *work(ii,nb+k) );
                        blasf77_zdscal( &n, &remax, work(0,nb+k), &ione );
                    }
                    lapackf77_zlacpy( "F", &n, &nb2, work(0,nb+iv), &n, VR(0,ki), &ldvr );
                    iv = nb;
                    timer_start( time_trsv );
                }
                else {
                    iv -= 1;
                }
            } // blocked back-transform

            // Restore the original diagonal elements of T.
            for( k=0; k <= ki - 1; ++k ) {
                *T(k,k) = *work(k,0);
            }

            is -= 1;
        }
    }
    timer_stop( time_trsv );

    timer_stop( time_total );
    timer_printf( "trevc trsv %.4f, gemm %.4f, gemv %.4f, total %.4f\n",
                  time_trsv_sum, time_gemm_sum, time_gemv_sum, time_total );

    if ( leftv ) {
        // ============================================================
        // Compute left eigenvectors.
        // iv is index of column in current block.
        // Non-blocked version always uses iv=1;
        // blocked     version starts with iv=1, goes up to nb.
        // (Note the "0-th" column is used to store the original diagonal.)
        iv = 1;
        is = 0;
        for( ki=0; ki < n; ++ki ) {
            if ( somev ) {
                if ( ! select[ki] ) {
                    continue;
                }
            }
            smin = max( ulp*MAGMA_Z_ABS1( *T(ki,ki) ), smlnum );

            // --------------------------------------------------------
            // Complex left eigenvector
            *work(ki,iv) = c_one;

            // Form right-hand side.
            for( k = ki + 1; k < n; ++k ) {
                *work(k,iv) = -MAGMA_Z_CONJ( *T(ki,k) );
            }

            // Solve conjugate-transposed triangular system:
            // [ T(ki+1:n,ki+1:n) - T(ki,ki) ]**H * X = scale*work.
            for( k = ki + 1; k < n; ++k ) {
                *T(k,k) = *T(k, k) - *T(ki,ki);
                if ( MAGMA_Z_ABS1( *T(k,k) ) < smin ) {
                    *T(k,k) = MAGMA_Z_MAKE( smin, 0. );
                }
            }

            if ( ki < n-1 ) {
                n2 = n-ki-1;
                lapackf77_zlatrs( "Upper", "Conjugate transpose", "Non-unit", "Y",
                                  &n2, T(ki+1,ki+1), &ldt,
                                  work(ki+1,iv), &scale, rwork, info );
                *work(ki,iv) = MAGMA_Z_MAKE( scale, 0. );
            }

            // Copy the vector x or Q*x to VL and normalize.
            if ( ! over ) {
                // ------------------------------
                // no back-transform: copy x to VL and normalize
                n2 = n-ki;
                blasf77_zcopy( &n2, work(ki,iv), &ione, VL(ki,is), &ione );

                ii = blasf77_izamax( &n2, VL(ki,is), &ione ) + ki - 1;
                remax = 1. / MAGMA_Z_ABS1( *VL(ii,is) );
                blasf77_zdscal( &n2, &remax, VL(ki,is), &ione );

                for( k=0; k < ki; ++k ) {
                    *VL(k,is) = c_zero;
                }
            }
            else if ( version == 1 ) {
                // ------------------------------
                // version 1: back-transform each vector with GEMV, Q*x.
                if ( ki < n-1 ) {
                    n2 = n-ki-1;
                    blasf77_zgemv( "n", &n, &n2, &c_one,
                                   VL(0,ki+1), &ldvl,
                                   work(ki+1,iv), &ione,
                                   work(ki,  iv), VL(0,ki), &ione );
                }
                ii = blasf77_izamax( &n, VL(0,ki), &ione ) - 1;
                remax = 1. / MAGMA_Z_ABS1( *VL(ii,ki) );
                blasf77_zdscal( &n, &remax, VL(0,ki), &ione );
            }
            else if ( version == 2 ) {
                // ------------------------------
                // version 2: back-transform block of vectors with GEMM
                // zero out above vector
                // could go from (ki+1)-NV+1 to ki
                for( k=0; k < ki; ++k ) {
                    *work(k,iv) = c_zero;
                }

                // Columns 1:iv of work are valid vectors.
                // When the number of vectors stored reaches nb,
                // or if this was last vector, do the GEMM
                if ( (iv == nb) || (ki == n-1) ) {
                    n2 = n-(ki+1)+iv;
                    blasf77_zgemm( "n", "n", &n, &iv, &n2, &c_one,
                                   VL(0,ki-iv+1), &ldvl,
                                   work(ki-iv+1,1   ), &n, &c_zero,
                                   work(0,      nb+1), &n );
                    // normalize vectors
                    for( k=1; k <= iv; ++k ) {
                        ii = blasf77_izamax( &n, work(0,nb+k), &ione ) - 1;
                        remax = 1. / MAGMA_Z_ABS1( *work(ii,nb+k) );
                        blasf77_zdscal( &n, &remax, work(0,nb+k), &ione );
                    }
                    lapackf77_zlacpy( "F", &n, &iv, work(0,nb+1), &n, VL(0,ki-iv+1), &ldvl );
                    iv = 1;
                }
                else {
                    iv += 1;
                }
            } // blocked back-transform

            // Restore the original diagonal elements of T.
            for( k = ki + 1; k < n; ++k ) {
                *T(k,k) = *work(k,0);
            }

            is += 1;
        }
    }
    
    return *info;
}  // End of ZTREVC
