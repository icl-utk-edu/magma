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
