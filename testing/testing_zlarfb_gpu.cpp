/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> c d s
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <algorithm>  // std::swap

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlarfb_gpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    // constants
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    
    // local variables
    magma_int_t M, N, K, size, ldc, ldv, ldt, ldw, ldw2, nv;
    magma_int_t ISEED[4] = {0,0,0,1};
    double Cnorm, error, work[1];
    int status = 0;
    
    // test all combinations of input parameters
    magma_side_t   side  [] = { MagmaLeft,       MagmaRight    };
    magma_trans_t  trans [] = { Magma_ConjTrans, MagmaNoTrans  };
    magma_direct_t direct[] = { MagmaForward,    MagmaBackward };
    magma_storev_t storev[] = { MagmaColumnwise, MagmaRowwise  };

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%%   M     N     K   side   trans   direct   storev   ||R||_F / ||HC||_F\n");
    printf("%%=======================================================================\n");
    for (int itest = 0; itest < opts.ntest; ++itest) {
      M = opts.msize[itest];
      N = opts.nsize[itest];
      K = opts.ksize[itest];
      for (int iside = 0; iside < 2; ++iside) {
      for (int itran = 0; itran < 2; ++itran) {
      for (int idir  = 0; idir  < 2; ++idir ) {
      for (int istor = 0; istor < 2; ++istor) {
        for (int iter = 0; iter < opts.niter; ++iter) {
            if ((side[iside] == MagmaLeft  && M < K) ||
                (side[iside] == MagmaRight && N < K))
            {
                printf( "%5lld %5lld %5lld   %4c   skipping because zlarfb requires M >= K (left) or N >= K (right)\n",
                        (long long) M, (long long) N, (long long) K,
                        lapacke_side_const(side[iside]) );
                continue;
            }

            ldc = magma_roundup( M, opts.align );  // multiple of 32 by default
            ldt = magma_roundup( K, opts.align );  // multiple of 32 by default
            ldw = (side[iside] == MagmaLeft ? N : M);
            ldw2 = min( M, N );
            // (ldv, nv) get swapped later if rowwise
            ldv = (side[iside] == MagmaLeft ? M : N);
            nv  = K;
            
            // Allocate memory for matrices
            magmaDoubleComplex *C, *R, *V, *T, *W;
            TESTING_CHECK( magma_zmalloc_cpu( &C, ldc*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &R, ldc*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &V, ldv*K ));
            TESTING_CHECK( magma_zmalloc_cpu( &T, ldt*K ));
            TESTING_CHECK( magma_zmalloc_cpu( &W, ldw*K ));
            
            magmaDoubleComplex_ptr dC, dV, dT, dW, dW2;
            TESTING_CHECK( magma_zmalloc( &dC,  ldc*N ));
            TESTING_CHECK( magma_zmalloc( &dV,  ldv*K ));
            TESTING_CHECK( magma_zmalloc( &dT,  ldt*K ));
            TESTING_CHECK( magma_zmalloc( &dW,  ldw*K ));
            TESTING_CHECK( magma_zmalloc( &dW2, ldw2*K ));
            
            // C is M x N.
            size = ldc*N;
            lapackf77_zlarnv( &ione, ISEED, &size, C );
            //printf( "C=" );  magma_zprint( M, N, C, ldc );
            
            // V is ldv x nv. See larfb docs for description.
            // if column-wise and left,  M x K
            // if column-wise and right, N x K
            // if row-wise and left,     K x M
            // if row-wise and right,    K x N
            size = ldv*nv;
            lapackf77_zlarnv( &ione, ISEED, &size, V );
            if ( storev[istor] == MagmaColumnwise ) {
                if ( direct[idir] == MagmaForward ) {
                    lapackf77_zlaset( MagmaUpperStr, &K, &K, &c_zero, &c_one, V, &ldv );
                }
                else {
                    lapackf77_zlaset( MagmaLowerStr, &K, &K, &c_zero, &c_one, &V[(ldv-K)], &ldv );
                }
            }
            else {
                // rowwise, swap V's dimensions
                std::swap( ldv, nv );
                if ( direct[idir] == MagmaForward ) {
                    lapackf77_zlaset( MagmaLowerStr, &K, &K, &c_zero, &c_one, V, &ldv );
                }
                else {
                    lapackf77_zlaset( MagmaUpperStr, &K, &K, &c_zero, &c_one, &V[(nv-K)*ldv], &ldv );
                }
            }
            //printf( "# ldv %lld, nv %lld\n", (long long) ldv, (long long) nv );
            //printf( "V=" );  magma_zprint( ldv, nv, V, ldv );
            
            // T is K x K, upper triangular for forward, and lower triangular for backward
            magma_int_t k1 = K-1;
            size = ldt*K;
            lapackf77_zlarnv( &ione, ISEED, &size, T );
            if ( direct[idir] == MagmaForward ) {
                lapackf77_zlaset( MagmaLowerStr, &k1, &k1, &c_zero, &c_zero, &T[1], &ldt );
            }
            else {
                lapackf77_zlaset( MagmaUpperStr, &k1, &k1, &c_zero, &c_zero, &T[1*ldt], &ldt );
            }
            //printf( "T=" );  magma_zprint( K, K, T, ldt );
            
            magma_zsetmatrix( M,   N,  C, ldc, dC, ldc, opts.queue );
            magma_zsetmatrix( ldv, nv, V, ldv, dV, ldv, opts.queue );
            magma_zsetmatrix( K,   K,  T, ldt, dT, ldt, opts.queue );
            
            lapackf77_zlarfb( lapack_side_const( side[iside] ), lapack_trans_const( trans[itran] ),
                              lapack_direct_const( direct[idir] ), lapack_storev_const( storev[istor] ),
                              &M, &N, &K,
                              V, &ldv, T, &ldt, C, &ldc, W, &ldw );
            //printf( "HC=" );  magma_zprint( M, N, C, ldc );
            
            if ( opts.version == 1 ) {
                magma_zlarfb_gpu( side[iside], trans[itran], direct[idir], storev[istor],
                                  M, N, K,
                                  dV, ldv, dT, ldt, dC, ldc, dW, ldw, opts.queue );
            }
            else {
                magma_zlarfb_gpu_gemm( side[iside], trans[itran], direct[idir], storev[istor],
                                       M, N, K,
                                       dV, ldv, dT, ldt, dC, ldc, dW, ldw, dW2, ldw2, opts.queue );
            }
            magma_zgetmatrix( M, N, dC, ldc, R, ldc, opts.queue );
            //printf( "dHC=" );  magma_zprint( M, N, R, ldc );
            
            // compute relative error |HC_magma - HC_lapack| / |HC_lapack|
            size = ldc*N;
            blasf77_zaxpy( &size, &c_neg_one, C, &ione, R, &ione );
            Cnorm = lapackf77_zlange( "Fro", &M, &N, C, &ldc, work );
            error = lapackf77_zlange( "Fro", &M, &N, R, &ldc, work ) / Cnorm;
            
            printf( "%5lld %5lld %5lld   %4c   %5c   %6c   %6c   %8.2e   %s\n",
                    (long long) M, (long long) N, (long long) K,
                    lapacke_side_const(side[iside]),
                    lapacke_trans_const(trans[itran]),
                    lapacke_direct_const(direct[idir]),
                    lapacke_storev_const(storev[istor]),
                   error, (error < tol ? "ok" : "failed") );
            status += ! (error < tol);
            
            magma_free_cpu( C );
            magma_free_cpu( R );
            magma_free_cpu( V );
            magma_free_cpu( T );
            magma_free_cpu( W );
            
            magma_free( dC  );
            magma_free( dV  );
            magma_free( dT  );
            magma_free( dW  );
            magma_free( dW2 );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }}}}
      printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
