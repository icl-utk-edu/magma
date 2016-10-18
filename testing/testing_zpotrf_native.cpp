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
#include "../control/magma_internal.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"
#include "cusolverDn.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_Rmagma;
    magmaDoubleComplex_ptr d_A;
    magma_int_t N, n2, lda, ldda, info;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      Anorm, magma_error, work[1];
    int status = 0;
    magma_int_t info_magma = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    printf("%% If running lapack (option --lapack), MAGMA error is computed \n"
               "%% relative to CPU result.\n\n");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% N     CPU Gflop/s (sec)   MAGMA Gflop/s (sec)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%==========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = N;
            n2  = lda*N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZPOTRF( N ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Rmagma, n2     ));
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N ));
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hpd( N, h_A, lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            magma_time = magma_wtime();
            magma_zpotrf_native( opts.uplo, N, d_A, ldda, opts.queue, &info_magma );
            magma_time = magma_wtime() - magma_time;
            magma_perf = gflops / magma_time;
            if (info_magma != 0) {
                printf("magma_zpotrf_native returned error %lld: %s.\n",
                       (long long) info_magma, magma_strerror( info_magma ));
            }
            magma_zgetmatrix( N, N, d_A, ldda, h_Rmagma, lda, opts.queue );
            
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
                Anorm = lapackf77_zlange("f", &N, &N, h_A, &lda, work);
                
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_Rmagma, &ione);
                magma_error = lapackf77_zlange("f", &N, &N, h_Rmagma, &lda, work) / Anorm;
                bool okay = (magma_error < tol);

                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     %8.2e %s\n",
                       (long long) N, cpu_perf, cpu_time, magma_perf, magma_time, 
                       magma_error, (okay ? "ok" : "failed") );
                status += ! okay;
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n", (long long) N, magma_perf, magma_time);
            }
            magma_free_cpu( h_A );
            magma_free_cpu( h_Rmagma );
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
