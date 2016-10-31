/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> s d c
*/

#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

/**
    Check the results following the LAPACK's [zcds]drvbd routine.
    A is factored as A = U diag(S) VT and the following 4 tests computed:
    (1)    | A - U diag(S) VT | / ( |A| max(m,n) )
    (2)    | I - U^H U   | / ( m )
    (3)    | I - VT VT^H | / ( n )
    (4)    S contains min(m,n) nonnegative values in decreasing order.
           (Return 0 if true, 1 if false.)
    
    If check is false, skips (1) - (3), but always does (4).
    ********************************************************************/
extern "C"
void check_zgesvd(
    magma_int_t check,
    magma_vec_t jobu,
    magma_vec_t jobv,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,  magma_int_t lda,
    double *S,
    magmaDoubleComplex *U,  magma_int_t ldu,
    magmaDoubleComplex *VT, magma_int_t ldv,
    double result[4] )
{
    double unused[1];
    const magma_int_t izero = 0;
    double eps = lapackf77_dlamch( "E" );
    
    if ( jobu == MagmaNoVec ) {
        U = NULL;
    }
    if ( jobv == MagmaNoVec ) {
        VT = NULL;
    }
    
    // -1 indicates check not done
    result[0] = -1;
    result[1] = -1;
    result[2] = -1;
    result[3] = -1;
    
    magma_int_t min_mn = min(m, n);
    magma_int_t n_u  = (jobu == MagmaAllVec ? m : min_mn);
    magma_int_t m_vt = (jobv == MagmaAllVec ? n : min_mn);
    
    assert( lda >= m );
    assert( ldu >= m );
    assert( ldv >= m_vt );
    
    if ( check ) {
        // zbdt01 needs m+n
        // zunt01 prefers n*(n+1) to check U; m*(m+1) to check V
        magma_int_t lwork_err = m+n;
        if ( U != NULL ) {
            lwork_err = max( lwork_err, n_u*(n_u+1) );
        }
        if ( VT != NULL ) {
            lwork_err = max( lwork_err, m_vt*(m_vt+1) );
        }
        magmaDoubleComplex *work_err;
        TESTING_CHECK( magma_zmalloc_cpu( &work_err, lwork_err ));
        
        // zbdt01 and zunt01 need max(m,n), depending
        double *rwork_err;
        TESTING_CHECK( magma_dmalloc_cpu( &rwork_err, max(m,n) ));
        
        if ( U != NULL && VT != NULL ) {
            // since KD=0 (3rd arg), E is not referenced so pass unused (9th arg)
            lapackf77_zbdt01( &m, &n, &izero, A, &lda,
                              U, &ldu, S, unused, VT, &ldv,
                              work_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[0] );
        }
        if ( U != NULL ) {
            lapackf77_zunt01( "Columns", &m,  &n_u, U,  &ldu, work_err, &lwork_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[1] );
        }
        if ( VT != NULL ) {
            lapackf77_zunt01( "Rows",    &m_vt, &n, VT, &ldv, work_err, &lwork_err,
                              #ifdef COMPLEX
                              rwork_err,
                              #endif
                              &result[2] );
        }
        
        result[0] *= eps;
        result[1] *= eps;
        result[2] *= eps;
        
        magma_free_cpu( work_err );
        magma_free_cpu( rwork_err );
    }
    
    // check S is sorted
    result[3] = 0.;
    for (int j=0; j < min_mn-1; j++) {
        if ( S[j] < S[j+1] )
            result[3] = 1.;
        if ( S[j] < 0. )
            result[3] = 1.;
    }
    if (min_mn > 1 && S[min_mn-1] < 0.) {
        result[3] = 1.;
    }
}
