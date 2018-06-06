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
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           gpu_error, cpu_error, error, Anorm, work[1];
    magmaDoubleComplex  c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_A2, *h_B, *h_X, *h_R, *tau, *h_work, tmp[1], unused[1];
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t M, N, size, nrhs, lda, ldb, ldda, lddb, min_mn, max_mn, nb, info;
    magma_int_t lworkgpu, lhwork, lhwork2;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    opts.parse_opts( argc, argv );
 
    int status = 0;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs = opts.nrhs;
    
    printf("%%                                                           ||b-Ax|| / (N||A||)   ||dx-x||/(N||A||)\n");
    printf("%%   M     N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   CPU        GPU                         \n");
    printf("%%==================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            if ( M < N ) {
                printf( "%5lld %5lld %5lld   skipping because M < N is not yet supported.\n", (long long) M, (long long) N, (long long) nrhs );
                continue;
            }
            min_mn = min(M, N);
            max_mn = max(M, N);
            lda    = M;
            ldb    = max_mn;
            size   = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( max_mn, opts.align );  // multiple of 32 by default
            nb     = magma_get_zgeqrf_nb( M, N );
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRS( M, N, nrhs )) / 1e9;
            
            lworkgpu = (M - N + nb)*(nrhs + nb) + nrhs*nb;
            
            // query for workspace size
            lhwork = -1;
            lapackf77_zgeqrf( &M, &N, unused, &M, unused, tmp, &lhwork, &info );
            lhwork2 = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
            
            lhwork = -1;
            lapackf77_zunmqr( MagmaLeftStr, Magma_ConjTransStr,
                              &M, &nrhs, &min_mn,
                              unused, &lda, unused,
                              unused, &ldb,
                              tmp, &lhwork, &info);
            lhwork = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
            lhwork = max( max( lhwork, lhwork2 ), lworkgpu );
            
            TESTING_CHECK( magma_zmalloc_cpu( &tau,    min_mn    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    lda*N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A2,   lda*N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,    ldb*nrhs  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X,    ldb*nrhs  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_R,    ldb*nrhs  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_work, lhwork    ));
            
            TESTING_CHECK( magma_zmalloc( &d_A,    ldda*N    ));
            TESTING_CHECK( magma_zmalloc( &d_B,    lddb*nrhs ));
            
            /* Initialize the matrices */
            magma_generate_matrix( opts, M, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            
            // make random RHS
            size = M*nrhs;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            // make consistent RHS
            //size = N*nrhs;
            //lapackf77_zlarnv( &ione, ISEED, &size, h_X );
            //blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
            //               &c_one,  h_A, &lda,
            //                        h_X, &ldb,
            //               &c_zero, h_B, &ldb );
            //lapackf77_zlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( M, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            gpu_time = magma_wtime();
            magma_zgels3_gpu( MagmaNoTrans, M, N, nrhs, d_A, ldda,
                              d_B, lddb, h_work, lworkgpu, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgels3_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            // Get the solution in h_X
            magma_zgetmatrix( N, nrhs, d_B, lddb, h_X, ldb, opts.queue );
            
            // compute the residual
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A, &lda,
                                       h_X, &ldb,
                           &c_one,     h_R, &ldb);
            Anorm = lapackf77_zlange("f", &M, &N, h_A, &lda, work);
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            lapackf77_zlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_X, &ldb );
            
            cpu_time = magma_wtime();
            lapackf77_zgels( MagmaNoTransStr, &M, &N, &nrhs,
                             h_A, &lda, h_X, &ldb, h_work, &lhwork, &info);
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_zgels returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A2, &lda,
                                       h_X,  &ldb,
                           &c_one,     h_B,  &ldb);
            
            cpu_error = lapackf77_zlange("f", &M, &nrhs, h_B, &ldb, work) / (min_mn*Anorm);
            gpu_error = lapackf77_zlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            // error relative to LAPACK
            size = M*nrhs;
            blasf77_zaxpy( &size, &c_neg_one, h_B, &ione, h_R, &ione );
            error = lapackf77_zlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            printf("%5lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e",
                   (long long) M, (long long) N, (long long) nrhs,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, cpu_error, gpu_error, error );
            
            if ( M == N ) {
                printf( "   %s\n", (gpu_error < tol && error < tol ? "ok" : "failed"));
                status += ! (gpu_error < tol && error < tol);
            }
            else {
                printf( "   %s\n", (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }

            magma_free_cpu( tau    );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_A2   );
            magma_free_cpu( h_B    );
            magma_free_cpu( h_X    );
            magma_free_cpu( h_R    );
            magma_free_cpu( h_work );
            
            magma_free( d_A    );
            magma_free( d_B    );
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
