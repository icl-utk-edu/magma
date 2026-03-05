/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Ahmad Abdelfattah
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrs_gpu
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_LU, *h_Bmagma, *h_X, *h_Blapack;
    magmaDoubleComplex_ptr d_LU, d_B;
    magma_int_t *ipiv;
    magma_int_t N, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs = opts.nrhs;

    printf("%% transA = %s\n", lapack_trans_const(opts.transA) );
    printf("%%   N  NRHS   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb   = ldda;
            gflops = FLOPS_ZGETRS( N, nrhs ) / 1e9;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,  lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_LU, lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Bmagma, ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Blapack, ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, ldb*nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));

            TESTING_CHECK( magma_zmalloc( &d_LU, ldda*N    ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs ));

            /* Initialize the matrices */
            sizeA = lda*N;
            sizeB = ldb*nrhs;

            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_Bmagma );
            lapackf77_zlacpy(MagmaFullStr, &N, &nrhs, h_Bmagma, &ldb, h_Blapack, &ldb);

            /* LU factorization of A */
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_LU, &lda );
            lapackf77_zgetrf( &N, &N, h_LU, &lda, ipiv, &info );
            magma_zsetmatrix( N, N, h_LU, lda, d_LU, ldda, opts.queue );

            magma_zsetmatrix( N, nrhs, h_Bmagma, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            if( opts.version == 1 ) {
                // top-level interface, auto-selects between hybrid/native modes
                // based on internal tuning
                gpu_time = magma_wtime();
                magma_zgetrs_gpu(opts.transA, N, nrhs, d_LU, ldda, ipiv, d_B, lddb, &info);
                gpu_time = magma_wtime() - gpu_time;
            }
            else if( opts.version == 2 || opts.version == 3) {
                // expert API, forcing hybrid or native mode
                // version 2 ==> hybrid, otherwise native
                magma_mode_t mode = (opts.version == 2) ? MagmaHybrid : MagmaNative;

                void *host_work=NULL, *device_work=NULL;
                magma_int_t lwork_host[1]   = {-1};
                magma_int_t lwork_device[1] = {-1};

                // query workspace
                magma_zgetrs_expert_gpu_work(
                    opts.transA, N, nrhs, NULL, ldda, NULL, NULL, lddb, &info,
                    mode, NULL, lwork_host, NULL, lwork_device, opts.queue );

                if(lwork_host[0] > 0) {
                    magma_malloc_cpu( &host_work, lwork_host[0] );
                }

                if(lwork_device[0] > 0) {
                    magma_malloc( &device_work, lwork_device[0] );
                }

                // time the main call only
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zgetrs_expert_gpu_work(
                    opts.transA, N, nrhs, d_LU, ldda, ipiv, d_B, lddb, &info,
                    mode, host_work, lwork_host,  device_work, lwork_device, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                if( host_work   != NULL ) magma_free_cpu( host_work );
                if( device_work != NULL ) magma_free( device_work );

            }
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgetrs_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_B, lddb, h_X, ldb, opts.queue );

            Anorm = lapackf77_zlange("I", &N, &N,    h_A, &lda, work);
            Xnorm = lapackf77_zlange("I", &N, &nrhs, h_X, &ldb, work);

            blasf77_zgemm( lapack_trans_const(opts.transA), MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_Bmagma, &ldb);

            Rnorm = lapackf77_zlange("I", &N, &nrhs, h_Bmagma, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status += ! (error < tol);

            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgetrs( lapack_trans_const(opts.transA), &N, &nrhs, h_LU, &lda, ipiv, h_Blapack, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;

                if (info != 0) {
                    printf("lapackf77_zgetrs returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }

                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                        error, (error < tol ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, gpu_perf, gpu_time*1000.,
                        error, (error < tol ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_LU );
            magma_free_cpu( h_Bmagma );
            magma_free_cpu( h_Blapack );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );

            magma_free( d_LU );
            magma_free( d_B );
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
