/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

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
   -- Testing zcgeqrsv
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time, gpu_perfd, gpu_perfs;
    double          error, gpu_error, cpu_error, Anorm, work[1];
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_A2, *h_B, *h_X, *h_R, unused[1];
    magmaDoubleComplex_ptr d_A, d_B, d_X, d_T;
    magmaFloatComplex  *d_SA, *d_SB;
    magmaDoubleComplex *h_workd, *tau, tmp[1];
    magmaFloatComplex  *h_works;
    magma_int_t lda,  ldb, lhwork, lworkgpu;
    magma_int_t ldda, lddb, lddx;
    magma_int_t M, N, nrhs, qrsv_iters, info, size, min_mn, max_mn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    printf("%% Epsilon(double): %8.6e\n"
           "%% Epsilon(single): %8.6e\n\n",
           lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon") );
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    nrhs = opts.nrhs;
    
    printf("%%                   CPU Gflop/s   GPU  Gflop/s                         |b-Ax|| / (N||A||)   ||dx-x||/(N||A||)\n");
    printf("%%   M     N  NRHS    double        double    single     mixed   Iter   CPU        GPU                        \n");
    printf("%%============================================================================================================\n");
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
            ldda   = magma_roundup( M,      opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( max_mn, opts.align );  // multiple of 32 by default
            lddx   = magma_roundup( N,      opts.align );  // multiple of 32 by default
            nb     = max( magma_get_zgeqrf_nb( M, N ),
                          magma_get_cgeqrf_nb( M, N ) );
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRS( M, N, nrhs )) / 1e9;
            
            lworkgpu = (M - N + nb)*(nrhs + nb) + nrhs*nb;
            
            // query for workspace size
            lhwork = -1;
            lapackf77_zgels( MagmaNoTransStr, &M, &N, &nrhs,
                             unused, &lda,
                             unused, &ldb,
                             tmp, &lhwork, &info );
            lhwork = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
            lhwork = max( lhwork, lworkgpu );
            
            TESTING_CHECK( magma_zmalloc_cpu( &tau,     min_mn   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,     lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A2,    lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,     ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X,     ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_R,     ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_workd, lhwork   ));
            h_works = (magmaFloatComplex*)h_workd;
            
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N      ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs   ));
            TESTING_CHECK( magma_zmalloc( &d_X, lddx*nrhs   ));
            TESTING_CHECK( magma_zmalloc( &d_T, ( 2*min_mn + magma_roundup( N, 32 ) )*nb ));
            
            /* Initialize the matrices */
            magma_generate_matrix( opts, M, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            
            // make random RHS
            size = ldb*nrhs;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            magma_zsetmatrix( M, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( M, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            //=====================================================================
            //              Mixed Precision Iterative Refinement - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            magma_zcgeqrsv_gpu( M, N, nrhs,
                                d_A, ldda, d_B, lddb,
                                d_X, lddx, &qrsv_iters, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zcgeqrsv returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            // compute the residual
            magma_zgetmatrix( N, nrhs, d_X, lddx, h_X, ldb, opts.queue );
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A, &lda,
                                       h_X, &ldb,
                           &c_one,     h_R, &ldb);
            Anorm = lapackf77_zlange("f", &M, &N,    h_A, &lda, work);
            
            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_zsetmatrix( M, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( M, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            gpu_time = magma_wtime();
            magma_zgels_gpu( MagmaNoTrans, M, N, nrhs, d_A, ldda,
                             d_B, lddb, h_workd, lworkgpu, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfd = gflops / gpu_time;
            
            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magma_zsetmatrix( M, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( M, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            /* The allocation of d_SA and d_SB is done here to avoid
             * to double the memory used on GPU with zcgeqrsv */
            TESTING_CHECK( magma_cmalloc( &d_SA, ldda*N    ));
            TESTING_CHECK( magma_cmalloc( &d_SB, lddb*nrhs ));
            magmablas_zlag2c( M, N,    d_A, ldda, d_SA, ldda, opts.queue, &info );
            magmablas_zlag2c( N, nrhs, d_B, lddb, d_SB, lddb, opts.queue, &info );
            
            gpu_time = magma_wtime();
            magma_cgels_gpu( MagmaNoTrans, M, N, nrhs, d_SA, ldda,
                             d_SB, lddb, h_works, lhwork, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfs = gflops / gpu_time;
            magma_free( d_SA );
            magma_free( d_SB );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            lapackf77_zlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_X, &ldb );
            
            cpu_time = magma_wtime();
            lapackf77_zgels( MagmaNoTransStr, &M, &N, &nrhs,
                             h_A, &lda, h_X, &ldb, h_workd, &lhwork, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_zgels returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A2, &lda,
                                       h_X,  &ldb,
                           &c_one,     h_B,  &ldb );
            
            cpu_error = lapackf77_zlange("f", &M, &nrhs, h_B, &ldb, work) / (min_mn*Anorm);
            gpu_error = lapackf77_zlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            // error relative to LAPACK
            size = M*nrhs;
            blasf77_zaxpy( &size, &c_neg_one, h_B, &ione, h_R, &ione );
            error = lapackf77_zlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            printf("%5lld %5lld %5lld   %7.2f       %7.2f   %7.2f   %7.2f   %4lld   %8.2e   %8.2e   %8.2e   %s\n",
                   (long long) M, (long long) N, (long long) nrhs,
                   cpu_perf, gpu_perfd, gpu_perfs, gpu_perf,
                   (long long) qrsv_iters,
                   cpu_error, gpu_error, error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            magma_free_cpu( tau  );
            magma_free_cpu( h_A  );
            magma_free_cpu( h_A2 );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_X  );
            magma_free_cpu( h_R  );
            magma_free_cpu( h_workd );
            
            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_X );
            magma_free( d_T );
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
