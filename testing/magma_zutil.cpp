/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Mark Gates

       Utilities for testing.
*/

#include <algorithm>  // sort

#include "magma_v2.h"
#include "magma_lapack.h"
#include "../control/magma_threadsetting.h"  // internal header, to work around MKL bug

#include "testings.h"

#define COMPLEX

#define A(i,j)  A[i + j*lda]

// --------------------
// Make a matrix symmetric/Hermitian.
// Makes diagonal real.
// Sets Aji = conj( Aij ) for j < i, that is, copy & conjugate lower triangle to upper triangle.
extern "C"
void magma_zmake_hermitian( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_Z_MAKE( MAGMA_Z_REAL( A(i,i) ), 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = MAGMA_Z_CONJ( A(i,j) );
        }
    }
}


// --------------------
// Make a matrix symmetric/Hermitian positive definite.
// Increases diagonal by N, and makes it real.
// Sets Aji = conj( Aij ) for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_zmake_hpd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_Z_MAKE( MAGMA_Z_REAL( A(i,i) ) + N, 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = MAGMA_Z_CONJ( A(i,j) );
        }
    }
}

#ifdef COMPLEX
// --------------------
// Make a matrix complex-symmetric
// Dose NOT make diagonal real.
// Sets Aji = Aij for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_zmake_symmetric( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        for( j=0; j < i; ++j ) {
            A(j,i) =  A(i,j);
        }
    }
}


// --------------------
// Make a matrix complex-symmetric positive definite.
// Increases diagonal by N. Does NOT make diagonal real.
// Sets Aji = Aij for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_zmake_spd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_Z_MAKE( MAGMA_Z_REAL( A(i,i) ) + N, MAGMA_Z_IMAG( A(i,i) ) );
        for( j=0; j < i; ++j ) {
            A(j,i) = A(i,j);
        }
    }
}
#endif


// --------------------
// MKL 11.1 has bug in multi-threaded zlanhe; use single thread to work around.
// MKL 11.2 corrects it for inf, one, max norm.
// MKL 11.2 still segfaults for Frobenius norm.
// See testing_zlanhe.cpp
extern "C"
double safe_lapackf77_zlanhe(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const magmaDoubleComplex *A, const magma_int_t *lda,
    double *work )
{
    #ifdef MAGMA_WITH_MKL
    // work around MKL bug in multi-threaded zlanhe
    magma_int_t la_threads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads( 1 );
    #endif
    
    double result = lapackf77_zlanhe( norm, uplo, n, A, lda, work );
    
    #ifdef MAGMA_WITH_MKL
    // end single thread to work around MKL bug
    magma_set_lapack_numthreads( la_threads );
    #endif
    
    return result;
}


// -----------------------------------------------------------------------------
// for sorting singular values
template< typename T >
bool greater( T a, T b )
{
    return (a > b);
}


// -----------------------------------------------------------------------------
// Generate random m x n test matrix A.
//
// matrix [in]      ID number of test matrix, from table below.
// m, n   [in]      matrix dimensions of A.
// iseed  [in,out]  random number generator seed. Elements in [0, 4095], last element odd.
// sigma  [out]     vector of size min(m,n). On output, contains singular values, if known, else set to -1.
// A      [out]     On output, test matrix A.
// lda    [in]      leading dimension of A.
//
// Requires LAPACK testing matrix generation library (-ltmglib).
// Matrix types from LAWN 41, table 11 (Test matrices for the singular value decomposition)
// plus some new matrices, marked with *, not in LAWN 41, Table 11
//
// type             |  Arithetic      |  Geometric  |  Clustered  |  Other
// -----------------+-----------------+-------------+-------------+-------------
// zero             |                 |             |             |  1
// identity         |                 |             |             |  2
// diagonal         |  3, 6, 7        |  4          |  5          |
// UDV              |  8, 11, 12, 19* |  9          |  10, 18     |
// random           |                 |             |             |  13, 14, 15, 0*
// random bidiag    |                 |             |             |  16*
// log random       |                 |             |             |  17*
//
// 6, 11, 14 entries are O( sqrt(overflow)  )
// 7, 12, 15 entries are O( sqrt(underflow) )
//
// 0 is random uniform in (0,1) using larnv [new]
// For 3-19, cond = 1/eps.
// arithmetic is sigma_i = 1 - (i - 1)/(n - 1)*(1 - 1/cond)
// geometric  is sigma_i = 1 - (cond)^{ -(i-1)/(n-1) }
// clustered:
//     10 is [ 1, 1/cond, ..., 1/cond ] -- deflates instantly (no D&C recursion)
//     18 is [ 1, ..., 1, 1/cond ] -- deflates quickly, but not instantly [new]
// 19 is arithmetic(5) -- sigma_i = 1 - floor((i-1)/5) / floor((n-1)/5) * (1 - 1/cond)
//                        like arithmetic, but repeats each entry 5 times [new]
// 17 is log random -- log(sigma_i) random uniform in (log(1/cond), log(1)) [new]

extern "C"
void magma_zgenerate_matrix(
    magma_int_t matrix,
    magma_int_t m, magma_int_t n,
    magma_int_t iseed[4],
    double* sigma,
    magmaDoubleComplex* A, magma_int_t lda )
{
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    const double d_zero    = MAGMA_D_ZERO;
    const double d_one     = MAGMA_D_ONE;
    const double d_neg_one = MAGMA_D_NEG_ONE;
    
    double ufl = lapackf77_dlamch( "safe min" );
    double ofl = lapackf77_dlamch( "overflow" );
    double ulp = lapackf77_dlamch( "precision" );  // Matlab eps
    double cond = 1 / ulp;
    magma_int_t min_mn = min( m, n );
    magma_int_t info = 0;
    
    // work for latms
    magmaDoubleComplex* work;
    TESTING_CHECK( magma_zmalloc_cpu( &work, 3 * max( m, n ) ));
    
    // determine scaling (Anorm)
    double Anorm = 1;
    switch (matrix) {
        case 6:
        case 11:
        case 14:
            Anorm = sqrt( ofl ) * ulp / max(m,n);
            break;
            
        case 7:
        case 12:
        case 15:
            Anorm = sqrt( ufl ) * min(m,n) / ulp;
            break;
            
        default:
            Anorm = 1;
    }
    
    // determine distribution of singular values (mode)
    magma_int_t mode = 0;
    switch (matrix) {
        case 3:
        case 6:
        case 7:
        case 8:
        case 11:
        case 12:
            // arithmetic dist., di = 1 - (i-1)/(n-1)*(1 - 1/cond)
            mode = 4;
            break;
        
        case 4:
        case 9:
            // geometric dist., di = cond^( -(i-1)/(n-1) )
            mode = 3;
            break;
        
        case 5:
        case 10:
            // clustered, d1 = 1, d{2...n} = 1/cond
            mode = 1;
            break;
        
        case 17:
            // sigma in (1/cond, 1); log(sigma) is uniformly distributed
            // in this case, sigma gets sorted below
            // this mode isn't in LAPACK's SVD tests, but latms supports it.
            mode = 5;
            break;
        
        case 18:
            // clustered, d(1...n-1) = 1, dn = 1/cond
            // this mode isn't in LAPACK's SVD tests, but latms supports it.
            mode = 2;
            break;
        
        case 19:
            // arithmetic(5): di = 1 - floor((i-1)/5) / floor((n-1)/5) * (1 - 1/cond)
            // this mode isn't in LAPACK's SVD tests, but latms supports it.
            mode = 0;
            int repeat = 5;
            int cnt = magma_ceildiv( min_mn, repeat );
            for (int i = 0; i < cnt; ++i) {
                for (int j = i*repeat; j < (i+1)*repeat && j < min_mn; ++j) {
                    sigma[j] = 1. - i/(cnt-1.)*(1. - 1./cond);
                }
            }
            break;
    }
    
    // generate matrix
    switch (matrix) {
        case 0: {
            // random; dist=1 uniform (0,1)
            // sigma is unknown; set to -1
            magma_int_t size = lda*n;
            lapackf77_zlarnv( &ione, iseed, &size, A );
            lapackf77_dlaset( "general", &min_mn, &ione, &d_neg_one, &d_neg_one, sigma, &min_mn );
            printf( "matrix %d: larnv\n", matrix );
            break;
        }
            
        case 1:
            // zero; sigma = 0
            lapackf77_zlaset( "general", &m, &n, &c_zero, &c_zero, A, &lda );
            lapackf77_dlaset( "general", &min_mn, &ione, &d_zero, &d_zero, sigma, &min_mn );
            printf( "matrix %d: laset( zero )\n", matrix );
            break;
            
        case 2:
            // identity; sigma = 1
            lapackf77_zlaset( "general", &m, &n, &c_zero, &c_one, A, &lda );
            lapackf77_dlaset( "general", &min_mn, &ione, &d_one, &d_one, sigma, &min_mn );
            printf( "matrix %d: laset( identity )\n", matrix );
            break;
            
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            // diagonal; sym. dist (-1, 1); non-sym matrix; kl = ku = 0
            #ifdef HAVE_LAPACK_TMG
            lapackf77_zlatms( &min_mn, &min_mn, "sym-dist", iseed, "non-sym",
                              sigma, &mode, &cond, &Anorm, &izero, &izero,
                              "no-pack", A, &lda, work, &info );
            printf( "matrix %d: latms diag, cond %.2e, Anorm %.2e\n",
                    matrix, cond, Anorm );
            #else
            fprintf( stderr, "For test matrices, add -DHAVE_LAPACK_TMG to make.inc and link with LAPACK -ltmglib\n" );
            exit(1);
            #endif
            break;
        
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 17:  // added
        case 18:  // added
        case 19:  // added
            // non-symmetric; sym. dist (-1, 1); non-sym matrix; kl = ku = 0
            #ifdef HAVE_LAPACK_TMG
            lapackf77_zlatms( &m, &n, "sym-dist", iseed, "non-sym",
                              sigma, &mode, &cond, &Anorm, &m, &n,
                              "no-pack", A, &lda, work, &info );
            printf( "matrix %d: latms non-symmetric, cond %.2e, Anorm %.2e\n",
                    matrix, cond, Anorm );
            #else
            fprintf( stderr, "For test matrices, add -DHAVE_LAPACK_TMG to make.inc and link with LAPACK -ltmglib\n" );
            exit(1);
            #endif
            break;
        
        case 13:
        case 14:
        case 15:
            // non-symmetric, latmr
            //lapackf77_zlatmr( &m, &n, "s", iseed, "n", work, 6, one, one,
            //                  "t", "n",
            //                  work(   min_mn+1 ), 1, one,
            //                  work( m+min_mn+1 ), 1, one,
            //                  "n", iwork, m, n, zero, Anorm,
            //                  "no", A, lda, iwork, info );
            fprintf( stderr, "random matrix (latmr) not yet implemented\n" );
            exit(1);
            break;
        
        case 16:
            // bidiagonal
            fprintf( stderr, "bidiagonal not yet implemented\n" );
            exit(1);
            break;
        
        default:
            fprintf( stderr, "Unknown matrix type %d\n", matrix );
            exit(1);
            break;
    }
    if (info != 0) {
        fprintf( stderr, "Error in %s: %d\n", __func__, info );
    }
    
    if (mode == 5) {
        // log random isn't previously sorted
        std::sort( sigma, &sigma[ min_mn ], greater<double> );
    }
    
    magma_free_cpu( work );
}
