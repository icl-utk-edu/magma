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
   -- Testing zhegst
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    // Constants
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;

    // Local variables
    real_Double_t gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_B, *h_R;
    double      Anorm, error, work[1];
    magma_int_t N, n2, lda, info;
    int status = 0;
    
    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% itype   N   CPU time (sec)   GPU time (sec)   |R|     \n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = N*lda;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,     lda*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,     lda*N ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,     lda*N ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            // todo: have different options for A and B
            magma_generate_matrix( opts, N, N, h_A, lda );
            magma_generate_matrix( opts, N, N, h_B, lda );
            magma_zpotrf( opts.uplo, N, h_B, lda, &info );
            if (info != 0) {
                printf("magma_zpotrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zhegst( opts.itype, opts.uplo, N, h_R, lda, h_B, lda, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zhegst returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zhegst( &opts.itype, lapack_uplo_const(opts.uplo),
                                  &N, h_A, &lda, h_B, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_zhegst returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                blasf77_zaxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = safe_lapackf77_zlanhe("f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_zlanhe("f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work )
                      / Anorm;
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%3lld   %5lld   %7.2f          %7.2f          %8.2e   %s\n",
                       (long long) opts.itype, (long long) N, cpu_time, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%3lld   %5lld     ---            %7.2f\n",
                       (long long) opts.itype, (long long) N, gpu_time );
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            
            magma_free_pinned( h_R );
            
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
