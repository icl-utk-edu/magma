/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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
   -- Testing zpotf2_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_A;
    magma_int_t N, n2, lda, ldda, info;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    double      Anorm, error, work[1];
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU Gflop/s (ms)    GPU Gflop/s (ms)    ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = N;
            n2  = lda*N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZPOTRF( N ) / 1e9;
            
            if ( N > 512 ) {
                printf( "%5lld   skipping because zpotf2 does not support N > 512\n", (long long) N );
                continue;
            }
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2     ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2     ));
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N ));
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );
            magma_zpotf2_gpu( opts.uplo, N, d_A, ldda, opts.queue, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zpotf2_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zpotrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_zgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                Anorm = lapackf77_zlange("f", &N, &N, h_A, &lda, work);
                error = lapackf77_zlange("f", &N, &N, h_R, &lda, work) / Anorm;
                
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) N, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                       error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (long long) N, gpu_perf, gpu_time*1000. );
            }
            magma_free_cpu( h_A );
            magma_free_pinned( h_R );
            magma_free( d_A );
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
