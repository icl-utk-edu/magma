/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Chongxiao Cao
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
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm
*/
int main( int argc, char** argv)
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dx(i_)     (dx + (i_))
    #define hA(i_, j_) (hA + (i_) + (j_)*lda)

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, device_perf, device_time, magma_perf, magma_time, cpu_perf=0, cpu_time=0;
    double          magma_error, device_error, normA, normx, normr, work[1];
    magma_int_t N, info;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magmaDoubleComplex *hA, *hb, *hx, *hxdevice, *hxmagma;
    magmaDoubleComplex_ptr dA, dx;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default; makes triangles nicely conditioned
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s, transA = %s, diag = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%   N  MAGMA Gflop/s (ms)   %s Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   %s error\n", g_platform_str, g_platform_str);
    printf("%%======================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            gflops = FLOPS_ZTRSM(opts.side, N, 1) / 1e9;
            lda    = N;
            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default

            TESTING_CHECK( magma_zmalloc_cpu( &hA,       lda*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &hb,       N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &hx,       N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &hxdevice, N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &hxmagma,  N     ));

            TESTING_CHECK( magma_zmalloc( &dA, ldda*N ));
            TESTING_CHECK( magma_zmalloc( &dx, N      ));

            /* Initialize the matrices */
            magma_generate_matrix( opts, N, N, hA, lda );

            // todo: setting to nan causes trsv to fail -- seems like a bug in cuBLAS?
            // set unused data to nan
            magma_int_t N_1 = N - 1;
            if (opts.uplo == MagmaLower)
                lapackf77_zlaset( "upper", &N_1, &N_1, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &hA[ 1*lda ], &lda );
            else
                lapackf77_zlaset( "lower", &N_1, &N_1, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &hA[ 1     ], &lda );

            // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
            // If diag == Unit, the diagonal is replaced; this is still well-conditioned.
            // First, brute force positive definiteness.
            for (int i = 0; i < N; ++i) {
                hA[ i + i*lda ] += N;
            }
            lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, hA, &lda, &info );
            assert( info == 0 );

            lapackf77_zlarnv( &ione, ISEED, &N, hb );
            blasf77_zcopy( &N, hb, &ione, hx, &ione );

            /* =====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( N, N, hA, lda, dA(0,0), ldda, opts.queue );
            magma_zsetvector( N, hx, 1, dx(0), 1, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ztrsv( opts.uplo, opts.transA, opts.diag,
                             N,
                             dA(0,0), ldda,
                             dx(0), 1, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_zgetvector( N, dx(0), 1, hxmagma, 1, opts.queue );

            /* =====================================================================
               Performs operation using CUBLAS/HIPBLAS
               =================================================================== */
            magma_zsetvector( N, hx, 1, dx(0), 1, opts.queue );

            device_time = magma_sync_wtime( opts.queue );
            magma_ztrsv( opts.uplo, opts.transA, opts.diag,
                         N,
                         dA(0,0), ldda,
                         dx(0), 1, opts.queue );
            device_time = magma_sync_wtime( opts.queue ) - device_time;
            device_perf = gflops / device_time;

            magma_zgetvector( N, dx(0), 1, hxdevice, 1, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrsv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               hA, &lda,
                               hx, &ione );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - Ax|| / (||A||*||x||)
            // error for CUBLAS
            normA = lapackf77_zlantr( "F",
                                      lapack_uplo_const(opts.uplo),
                                      lapack_diag_const(opts.diag),
                                      &N, &N, hA, &lda, work );

            normx = lapackf77_zlange( "F", &N, &ione, hxdevice, &ione, work );

            blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &N,
                           hA, &lda,
                           hxdevice, &ione );
            blasf77_zaxpy( &N, &c_neg_one, hb, &ione, hxdevice, &ione );
            normr = lapackf77_zlange( "F", &N, &ione, hxdevice, &N, work );
            device_error = normr / (normA*normx);

            blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &N,
                           hA, &lda,
                           hxmagma, &ione );
            blasf77_zaxpy( &N, &c_neg_one, hb, &ione, hxmagma, &ione );
            normr = lapackf77_zlange( "F", &N, &ione, hxmagma, &N, work );
            magma_error = normr / (normA*normx);

            bool okay = (magma_error < tol && device_error < tol);
            status += ! okay;
            if ( opts.lapack ) {
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                        (long long) N,
                        magma_perf, 1000.*magma_time,
                        device_perf, 1000.*device_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, device_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---  (  ---  )   %8.2e   %8.2e   %s\n",
                        (long long) N,
                        magma_perf, 1000.*magma_time,
                        device_perf, 1000.*device_time,
                        magma_error, device_error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( hA  );
            magma_free_cpu( hb  );
            magma_free_cpu( hx  );
            magma_free_cpu( hxdevice );
            magma_free_cpu( hxmagma );

            magma_free( dA );
            magma_free( dx );
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
