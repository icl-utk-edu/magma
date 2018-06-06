/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Stan Tomov

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    // Constants
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const double d_one     =  1;
    const double d_neg_one = -1;

    // Local variables
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           Anorm, error=0, error2=0;
    magmaDoubleComplex *h_A, *h_R, *tau, *dtau, *h_work, tmp[1];
    magmaDoubleComplex_ptr d_A;
    magmaDouble_ptr dwork;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% M     N     CPU Gflop/s (ms)    GPU Gflop/s (ms)    |R - Q^H*A|   |I - Q^H*Q|\n");
    printf("%%==============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_zgeqrf( &M, &N, tmp, &M, tmp, tmp, &lwork, &info );
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            
            TESTING_CHECK( magma_zmalloc_cpu( &tau,    min_mn ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_work, lwork  ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,    n2     ));
            
            TESTING_CHECK( magma_zmalloc( &d_A,    ldda*N ));
            TESTING_CHECK( magma_zmalloc( &dtau,   min_mn ));
            TESTING_CHECK( magma_dmalloc( &dwork,  min_mn ));
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, M, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
            
            // warmup
            if ( opts.warmup ) {
                magma_zgeqr2_gpu( M, N, d_A, ldda, dtau, dwork, opts.queue, &info );
                magma_zsetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );

            magma_zgeqr2_gpu( M, N, d_A, ldda, dtau, dwork, opts.queue, &info );

            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgeqr2_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the result, following zqrt01 except using the reduced Q.
               This works for any M,N (square, tall, wide).
               =================================================================== */
            if ( opts.check ) {
                magma_zgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );
                magma_zgetvector( min_mn, dtau, 1, tau, 1, opts.queue );
                
                magma_int_t ldq = M;
                magma_int_t ldr = min_mn;
                magmaDoubleComplex *Q, *R;
                double *cwork;
                TESTING_CHECK( magma_zmalloc_cpu( &Q,     ldq*min_mn ));  // M by K
                TESTING_CHECK( magma_zmalloc_cpu( &R,     ldr*N ));       // K by N
                TESTING_CHECK( magma_dmalloc_cpu( &cwork, min_mn ));
                
                // generate M by K matrix Q, where K = min(M,N)
                lapackf77_zlacpy( "Lower", &M, &min_mn, h_R, &lda, Q, &ldq );
                lapackf77_zungqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                assert( info == 0 );
                
                // copy K by N matrix R
                lapackf77_zlaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                lapackf77_zlacpy( "Upper", &min_mn, &N, h_R, &lda,        R, &ldr );
                
                // error = || R - Q^H*A || / (N * ||A||)
                blasf77_zgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                               &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                Anorm = lapackf77_zlange( "1", &M,      &N, h_A, &lda, cwork );
                error = lapackf77_zlange( "1", &min_mn, &N, R,   &ldr, cwork );
                if ( N > 0 && Anorm > 0 )
                    error /= (N*Anorm);
                
                // set R = I (K by K identity), then R = I - Q^H*Q
                // error = || I - Q^H*Q || / N
                lapackf77_zlaset( "Upper", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr );
                blasf77_zherk( "Upper", "Conj", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr );
                error2 = safe_lapackf77_zlanhe( "1", "Upper", &min_mn, R, &ldr, cwork );
                if ( N > 0 )
                    error2 /= N;
                
                magma_free_cpu( Q     );  Q     = NULL;
                magma_free_cpu( R     );  R     = NULL;
                magma_free_cpu( cwork );  cwork = NULL;
            }
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgeqrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) M, (long long) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) M, (long long) N, gpu_perf, 1000.*gpu_time );
            }
            
            if ( opts.check ) {
                bool okay = (error < tol) && (error2 < tol);
                status += ! okay;
                printf("   %8.2e      %8.2e   %s\n",
                       error, error2, (okay ? "ok" : "failed"));
            }
            else {
                printf("     ---  \n");
            }
            
            magma_free_cpu( tau    );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_work );
            
            magma_free_pinned( h_R   );
            
            magma_free( d_A   );
            magma_free( dtau  );
            magma_free( dwork );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
