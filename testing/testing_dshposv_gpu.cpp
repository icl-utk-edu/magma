/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_RESET   "\x1b[0m"

static inline bool begins( std::string const &str, std::string const &prefix )
{
    return (str.compare( 0, prefix.size(), prefix) == 0);
}

int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflopsF, gflopsS, gpu_perf, gpu_time;
    real_Double_t   gpu_perfdf, gpu_perfds;
    real_Double_t   gpu_perfsf, gpu_perfss;
    double          error, Rnorm, Anorm;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_B, *h_X;
    magmaDouble_ptr d_A,  d_B,  d_X, d_workd;
    magmaFloat_ptr  d_As, d_Bs,      d_works;
    double          *h_workd;
    magma_int_t lda, ldb, ldx;
    magma_int_t N, nrhs, posv_iter, info, size;
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

    if( !begins(opts.matrix, "poev") ) {
        printf(ANSI_COLOR_MAGENTA "%% WARNING: matrix option \"%s\" will be explicitly made SPD\n" ANSI_COLOR_RESET, opts.matrix.c_str());
        printf(ANSI_COLOR_MAGENTA "%% Use matrix option \"poev\" or \"poev_<distribution>\" to avoid this warning\n\n" ANSI_COLOR_RESET);
    }

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo));
    printf("%%    N NRHS   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  Iter   |b-Ax|/|A|\n");
    printf("%%====================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            ldb = ldx = lda = N;
            gflopsF = FLOPS_DPOTRF( N ) / 1e9;
            gflopsS = gflopsF + FLOPS_DPOTRS( N, nrhs ) / 1e9;

            TESTING_CHECK( magma_dmalloc_cpu( &h_A,     lda*N    ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_B,     ldb*nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_X,     ldx*nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_workd, N        ));

            TESTING_CHECK( magma_dmalloc( &d_A,     lda*N        ));
            TESTING_CHECK( magma_dmalloc( &d_B,     ldb*nrhs     ));
            TESTING_CHECK( magma_dmalloc( &d_X,     ldx*nrhs     ));
            TESTING_CHECK( magma_smalloc( &d_works, lda*(N+nrhs) + N ));  // an extra 'N' is required to store the diagonal
            TESTING_CHECK( magma_dmalloc( &d_workd, N*nrhs ));

            /* Initialize the matrix */
            magma_generate_matrix(opts, N, N, h_A, lda );
            if( !begins(opts.matrix, "poev") ) {
                magma_dmake_hpd( N, h_A, lda );
            }

            size = ldb * nrhs;
            lapackf77_dlarnv( &ione, ISEED, &size, h_B );

            magma_dsetmatrix( N, N,    h_A, lda, d_A, lda, opts.queue );
            magma_dsetmatrix( N, nrhs, h_B, ldb, d_B, ldb, opts.queue );

            //=====================================================================
            //              Mixed Precision Iterative Refinement - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_dshposv_gpu(
                    opts.uplo, N, nrhs,
                    d_A, lda,
                    d_B, ldb,
                    d_X, ldx,
                    &posv_iter, &info);
            }
            else if (opts.version == 2) {
                magma_dshposv_native(
                    opts.uplo, N, nrhs,
                    d_A, lda,
                    d_B, ldb,
                    d_X, ldx,
                    &posv_iter, &info);
            }

            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_dshposv returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            //                 Error Computation
            //=====================================================================
            magma_dgetmatrix( N, nrhs, d_X, ldx, h_X, ldx, opts.queue );

            Anorm = safe_lapackf77_dlansy( "I", lapack_uplo_const(opts.uplo), &N, h_A, &lda, h_workd);
            blasf77_dsymm( "L", lapack_uplo_const(opts.uplo), &N, &nrhs,
                           &c_one,     h_A, &lda,
                                       h_X, &ldx,
                           &c_neg_one, h_B, &ldb);
            Rnorm = lapackf77_dlange( "I", &N, &nrhs, h_B, &ldb, h_workd);
            error = Rnorm / (N * Anorm);

            //=====================================================================
            //                 Double Precision Factor
            //=====================================================================
            magma_dsetmatrix( N, N, h_A, lda, d_A, lda, opts.queue );

            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_dpotrf_gpu(opts.uplo, N, d_A, lda, &info);
            }
            else{
                magma_dpotrf_native(opts.uplo, N, d_A, lda, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfdf = gflopsF / gpu_time;
            if (info != 0) {
                printf("magma_dpotrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_dsetmatrix( N, N,    h_A, lda, d_A, lda, opts.queue );
            magma_dsetmatrix( N, nrhs, h_B, ldb, d_B, ldb, opts.queue );

            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_dpotrf_gpu(opts.uplo, N, d_A, lda, &info);
            }
            else{
                magma_dpotrf_native(opts.uplo, N, d_A, lda, &info);
            }
            magma_dpotrs_gpu(opts.uplo, N, nrhs, d_A, lda, d_B, ldb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfds = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_dpotrs returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            //                 Single Precision Factor
            //=====================================================================
            d_As = d_works;
            d_Bs = d_works + lda*N;
            magma_dsetmatrix( N, N,    h_A, lda, d_A, lda, opts.queue );
            magma_dsetmatrix( N, nrhs, h_B, ldb, d_B, ldb, opts.queue );
            magmablas_dlag2s( N, N,    d_A, lda, d_As, N, opts.queue, &info );
            magmablas_dlag2s( N, nrhs, d_B, ldb, d_Bs, N, opts.queue, &info );

            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_spotrf_gpu(opts.uplo, N, d_As, N, &info);
            }
            else{
                magma_spotrf_native(opts.uplo, N, d_As, N, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfsf = gflopsF / gpu_time;
            if (info != 0) {
                printf("magma_spotrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magmablas_dlag2s(N, N,    d_A, lda, d_As, N, opts.queue, &info );
            magmablas_dlag2s(N, nrhs, d_B, ldb, d_Bs, N, opts.queue, &info );

            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_spotrf_gpu(opts.uplo, N, d_As, N, &info);
            }
            else{
                magma_spotrf_native(opts.uplo, N, d_As, N, &info);
            }
            magma_spotrs_gpu(opts.uplo, N, nrhs, d_As, N, d_Bs, N, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfss = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_spotrs returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            printf(" %5lld %4lld   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f    %4lld   %8.2e   %s\n",
                   (long long) N, (long long) nrhs,
                   gpu_perfdf, gpu_perfds, gpu_perfsf, gpu_perfss, gpu_perf,
                   (long long) posv_iter, error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( h_workd );

            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_X );
            magma_free( d_works );
            magma_free( d_workd );
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
