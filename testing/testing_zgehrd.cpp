/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates

       @precisions normal z -> s d c
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

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgehrd
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *h_Q, *h_work, *tau, *twork, *T;
    magmaDoubleComplex_ptr dT;
    #ifdef COMPLEX
    double      *rwork;
    #endif
    double      result[2];
    magma_int_t N, n2, lda, nb, lwork, ltwork, info;
    magma_int_t ione     = 1;
    int status = 0;
        
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    double eps = lapackf77_dlamch( "E" );
    
    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% version %lld, ngpu = %lld\n", (long long) opts.version, (long long) abs_ngpu);
    
    printf("%%   N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |A-QHQ^H|/N|A|   |I-QQ^H|/N\n");
    printf("%%==========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_zgehrd_nb( N );
            // magma needs larger workspace than lapack, esp. multi-gpu verison
            lwork  = N*nb;
            if (opts.ngpu != 1) {
                lwork += N*nb*abs_ngpu;
            }
            gflops = FLOPS_ZGEHRD( N ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    n2    ));
            TESTING_CHECK( magma_zmalloc_cpu( &tau,    N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &T,      nb*N  ));  // for multi GPU
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,    n2    ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_work, lwork ));
            
            TESTING_CHECK( magma_zmalloc( &dT,     nb*N  ));  // for single GPU
            
            /* Initialize the matrices */
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 1 ) {
                if ( opts.ngpu == 1 ) {
                    magma_zgehrd( N, ione, N, h_R, lda, tau, h_work, lwork, dT, &info );
                }
                else {
                    magma_zgehrd_m( N, ione, N, h_R, lda, tau, h_work, lwork, T, &info );
                }
            }
            else {
                // LAPACK-complaint arguments, no dT array
                printf( "magma_zgehrd2\n" );
                magma_zgehrd2( N, ione, N, h_R, lda, tau, h_work, lwork, &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgehrd returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                ltwork = 2*N*N;
                TESTING_CHECK( magma_zmalloc_pinned( &h_Q,   lda*N  ));
                TESTING_CHECK( magma_zmalloc_cpu( &twork, ltwork ));
                #ifdef COMPLEX
                TESTING_CHECK( magma_dmalloc_cpu( &rwork, N ));
                #endif
                
                lapackf77_zlacpy( MagmaFullStr, &N, &N, h_R, &lda, h_Q, &lda );
                for( int j = 0; j < N-1; ++j )
                    for( int i = j+2; i < N; ++i )
                        h_R[i+j*lda] = MAGMA_Z_ZERO;
                
                if ( opts.version == 1 ) {
                    if ( opts.ngpu != 1 ) {
                        magma_zsetmatrix( nb, N, T, nb, dT, nb, opts.queue );
                    }
                    magma_zunghr( N, ione, N, h_Q, lda, tau, dT, nb, &info );
                }
                else {
                    // for magma_zgehrd2, no dT array
                    lapackf77_zunghr( &N, &ione, &N, h_Q, &lda, tau, h_work, &lwork, &info );
                }
                if (info != 0) {
                    printf("magma_zunghr returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                    return -1;
                }
                lapackf77_zhst01( &N, &ione, &N,
                                  h_A, &lda, h_R, &lda,
                                  h_Q, &lda, twork, &ltwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  result );
                
                magma_free_pinned( h_Q   );
                magma_free_cpu( twork );
                #ifdef COMPLEX
                magma_free_cpu( rwork );
                #endif
                
                // lapack normalizes by eps
                result[0] *= eps;
                result[1] *= eps;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgehrd( &N, &ione, &N, h_A, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgehrd returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                bool okay = (result[0] < tol) && (result[1] < tol);
                status += ! okay;
                printf("   %8.2e        %8.2e   %s\n",
                       result[0], result[1],
                       (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---             ---\n");
            }
            
            magma_free_cpu( h_A    );
            magma_free_cpu( tau    );
            magma_free_cpu( T      );
            
            magma_free_pinned( h_R    );
            magma_free_pinned( h_work );
            
            magma_free( dT     );
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
