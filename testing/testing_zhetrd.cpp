/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
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
   -- Testing zhetrd
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double           eps;
    magmaDoubleComplex *h_A, *h_R, *h_Q, *h_work, *work;
    magmaDoubleComplex *tau;
    double          *diag, *offdiag;
    double           result[2] = {0., 0.};
    magma_int_t N, lda, lwork, info, nb;
    magma_int_t ione     = 1;
    magma_int_t itwo     = 2;
    magma_int_t ithree   = 3;
    int status = 0;
    
    #ifdef COMPLEX
    double *rwork;
    #endif

    eps = lapackf77_dlamch( "E" );

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% N     CPU Gflop/s (sec)   GPU Gflop/s (sec)   |A-QHQ^H|/N|A|   |I-QQ^H|/N\n");
    printf("%%==========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            nb     = magma_get_zhetrd_nb(N);
            lwork  = N*nb;  /* We suppose the magma nb is bigger than lapack nb */
            gflops = FLOPS_ZHETRD( N ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,     lda*N ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_work,  lwork ));
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,     lda*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &tau,     N     ));
            TESTING_CHECK( magma_dmalloc_cpu( &diag,    N   ));
            TESTING_CHECK( magma_dmalloc_cpu( &offdiag, N-1 ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zhetrd( opts.uplo, N, h_R, lda, diag, offdiag,
                          tau, h_work, lwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zhetrd returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                TESTING_CHECK( magma_zmalloc_cpu( &h_Q,  lda*N ));
                TESTING_CHECK( magma_zmalloc_cpu( &work, 2*N*N ));
                #ifdef COMPLEX
                TESTING_CHECK( magma_dmalloc_cpu( &rwork, N ));
                #endif

                lapackf77_zlacpy( lapack_uplo_const(opts.uplo), &N, &N, h_R, &lda, h_Q, &lda );
                lapackf77_zungtr( lapack_uplo_const(opts.uplo), &N, h_Q, &lda, tau, h_work, &lwork, &info );
                
                lapackf77_zhet21( &itwo, lapack_uplo_const(opts.uplo), &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[0] );
                
                lapackf77_zhet21( &ithree, lapack_uplo_const(opts.uplo), &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[1] );
                result[0] *= eps;
                result[1] *= eps;
                
                magma_free_cpu( h_Q  );
                magma_free_cpu( work );
                #ifdef COMPLEX
                magma_free_cpu( rwork );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zhetrd( lapack_uplo_const(opts.uplo), &N, h_A, &lda, diag, offdiag, tau,
                                  h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zhetrd returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            } else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                printf("   %8.2e        %8.2e   %s\n", result[0], result[1],
                        ((result[0] < tol && result[1] < tol) ? "ok" : "failed") );
                status += ! (result[0] < tol && result[1] < tol);
            } else {
                printf("     ---             ---\n");
            }
            
            magma_free_pinned( h_R    );
            magma_free_pinned( h_work );
            
            magma_free_cpu( h_A     );
            magma_free_cpu( tau     );
            magma_free_cpu( diag    );
            magma_free_cpu( offdiag );
            
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
