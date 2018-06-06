/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgels
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           gpu_err1, gpu_err2, cpu_err1, cpu_err2, work[1];
    magmaDoubleComplex  c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_A2, *h_B, *h_B2, *h_R, *h_work, tmp[1], unused[1];
    magmaDoubleComplex *h_c, *h_d, *h_x, *h_c2, *h_d2, *h_x2;
    magma_int_t M, N, size, P, lda, ldb, nb, info;
    magma_int_t lhwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    opts.parse_opts( argc, argv );
 
    int status = 0;

    printf("%%                                                          ||c-Ax||/||d-Bx||  ||c-Ax||/||d-Bx||   \n");
    printf("%%   M     N    P   CPU Gflop/s (sec)   GPU Gflop/s (sec)          CPU                GPU          \n");
    printf("%%==================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            P = opts.ksize[itest];
            if ( !( P<=N && N <=M+P) ) {
                printf( "%5lld %5lld %5lld   skipping because we don't have P <= N <= M+P.\n", 
                        (long long) M, (long long) N, (long long) P );
                continue;
            }
            lda    = M;
            ldb    = P;
            nb     = magma_get_zgeqrf_nb( M, N );
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRS( M, N, P )) / 1e9;
            
            // query for workspace size
            lhwork = -1;
            lapackf77_zgglse( &M, &N, &P,
                              unused, &lda,
                              unused, &ldb,
                              unused, unused, unused,
                              tmp, &lhwork, &info );
            lhwork = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
            lhwork = max(lhwork, M*nb + P + min(M,N));
            lhwork = max(lhwork, 2*nb*nb );

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    lda*N     ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_A2,   lda*N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,    ldb*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B2,   ldb*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_R,    ldb*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_work, lhwork    ));
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_c , M  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_d , P  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_x , N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_c2, M  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_d2, P  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_x2, N  ));

            /* Initialize the matrices */
            magma_generate_matrix( opts, M, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            
            // make random RHS
            size = ldb*N;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaFullStr, &P, &N, h_B, &ldb, h_R , &ldb );
            lapackf77_zlacpy( MagmaFullStr, &P, &N, h_B, &ldb, h_B2, &ldb );

            lapackf77_zlarnv( &ione, ISEED, &M, h_c );
            lapackf77_zlarnv( &ione, ISEED, &P, h_d );
            lapackf77_zlarnv( &ione, ISEED, &N, h_x );
            lapackf77_zlacpy( MagmaFullStr, &M, &ione, h_c, &M, h_c2, &M );
            lapackf77_zlacpy( MagmaFullStr, &P, &ione, h_d, &P, h_d2, &P );
            lapackf77_zlacpy( MagmaFullStr, &N, &ione, h_x, &N, h_x2, &N );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgglse( M, N, P, 
                          h_A2, lda,
                          h_B2, ldb, 
                          h_c2, h_d2, h_x2,
                          h_work, lhwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgglse returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            // compute the residual
            lapackf77_zlacpy( MagmaFullStr, &M, &ione, h_c, &M, h_c2, &M );
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &ione, &N,
                           &c_neg_one, h_A, &lda,
                                       h_x2, &N,
                           &c_one,     h_c2, &M );
            gpu_err1 = lapackf77_zlange("f", &M, &ione, h_c2, &M, work); 

            lapackf77_zlacpy( MagmaFullStr, &P, &ione, h_d, &P, h_d2, &P );
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &P, &ione, &N,
                           &c_neg_one, h_B, &ldb,
                           h_x2, &N,
                           &c_one,     h_d2, &P );
            gpu_err2 = lapackf77_zlange("f", &P, &ione, h_d2, &P, work);

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            lapackf77_zlacpy( MagmaFullStr, &P, &N, h_B, &ldb, h_B2, &ldb );
            
            lapackf77_zlacpy( MagmaFullStr, &M, &ione, h_c, &M, h_c2, &M );
            lapackf77_zlacpy( MagmaFullStr, &P, &ione, h_d, &P, h_d2, &P );
            lapackf77_zlacpy( MagmaFullStr, &N, &ione, h_x, &N, h_x2, &N );

            cpu_time = magma_wtime();
            lapackf77_zgglse( &M, &N, &P,
                              h_A2, &lda, 
                              h_B2, &ldb,
                              h_c2, h_d2, h_x2, 
                              h_work, &lhwork, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_zgels returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            // compute the residual
            lapackf77_zlacpy( MagmaFullStr, &M, &ione, h_c, &M, h_c2, &M );
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &ione, &N,
                           &c_neg_one, h_A, &lda,
                           h_x2, &N,
                           &c_one,     h_c2, &M );
            cpu_err1 = lapackf77_zlange("f", &M, &ione, h_c2, &M, work);

            lapackf77_zlacpy( MagmaFullStr, &P, &ione, h_d, &P, h_d2, &P );
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &P, &ione, &N,
                           &c_neg_one, h_B, &ldb,
                           h_x2, &N,
                           &c_one,     h_d2, &P );
            cpu_err2 = lapackf77_zlange("f", &P, &ione, h_d2, &P, work);

            printf("%5lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)  %8.2e/%8.2e  %8.2e/%8.2e",
                   (long long) M, (long long) N, (long long) P,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, cpu_err1, cpu_err2, gpu_err1, gpu_err2);
            
            printf( "  %s\n", (0.9*gpu_err1 < cpu_err1 && 0.5*gpu_err2 < cpu_err2 ? "ok" : "failed"));
            status += ! (0.9*gpu_err1 < cpu_err1 && 0.9*gpu_err2 < cpu_err2);

            magma_free_cpu( h_A    );
            magma_free_pinned( h_A2   );
            magma_free_cpu( h_B    );
            magma_free_cpu( h_B2   );
            magma_free_cpu( h_R    );
            magma_free_cpu( h_work );
            
            magma_free_cpu( h_c    );
            magma_free_cpu( h_d    );
            magma_free_cpu( h_x    );
            magma_free_cpu( h_c2   );
            magma_free_cpu( h_d2   );
            magma_free_cpu( h_x2   );

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
