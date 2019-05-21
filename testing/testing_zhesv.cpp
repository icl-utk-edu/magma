/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Ichitaro Yamazaki
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

/* ================================================================================================== */

// Initialize matrix to random.
// This ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix(
    magma_opts &opts,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda )
{
    magma_int_t iseed_save[4];
    for (magma_int_t i = 0; i < 4; ++i) {
        iseed_save[i] = opts.iseed[i];
    }

    magma_generate_matrix( opts, m, n, A, lda );

    // restore iseed
    for (magma_int_t i = 0; i < 4; ++i) {
        opts.iseed[i] = iseed_save[i];
    }
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
double get_residual(
    magma_opts &opts,
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magmaDoubleComplex *x, magma_int_t ldx,
    magmaDoubleComplex *b, magma_int_t ldb)
{
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    
    // reset to original A
    init_matrix( opts, n, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_zhemv( lapack_uplo_const(uplo), &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    double norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_zlanhe( "Fro", lapack_uplo_const(uplo), &n, A, &lda, work );
    norm_r = lapackf77_zlange( "Fro", &n, &ione, b, &n, work );
    norm_x = lapackf77_zlange( "Fro", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_zprint( 1, n, b, 1 );
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhesv
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magmaDoubleComplex *h_A, *h_B, *h_X, *work, temp;
    real_Double_t   gflops, gpu_perf, gpu_time = 0.0, cpu_perf=0, cpu_time=0;
    double          error, error_lapack = 0.0;
    magma_int_t     *ipiv;
    magma_int_t     N, n2, lda, ldb, sizeB, lwork, info;
    magma_int_t     ione = 1;
    magma_int_t     ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            ldb    = N;
            lda    = N;
            n2     = lda*N;
            sizeB  = ldb*opts.nrhs;
            gflops = ( FLOPS_ZPOTRF( N ) + FLOPS_ZPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_A,  n2 ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_B,  sizeB ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_X,  sizeB ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                lwork = -1;
                lapackf77_zhesv(lapack_uplo_const(opts.uplo), &N, &opts.nrhs,
                                h_A, &lda, ipiv, h_X, &ldb, &temp, &lwork, &info);
                lwork = (magma_int_t)MAGMA_Z_REAL(temp);
                TESTING_CHECK( magma_zmalloc_cpu( &work, lwork ));

                init_matrix( opts, N, N, h_A, lda );
                lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
                lapackf77_zlacpy( MagmaFullStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );

                cpu_time = magma_wtime();
                lapackf77_zhesv(lapack_uplo_const(opts.uplo), &N, &opts.nrhs,
                                h_A, &lda, ipiv, h_X, &ldb, work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zhesv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                error_lapack = get_residual( opts, opts.uplo, N, opts.nrhs, h_A, lda, ipiv, h_X, ldb, h_B, ldb );

                magma_free_cpu( work );
            }
           
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( opts, N, N, h_A, lda );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_zlacpy( MagmaFullStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );

            magma_setdevice(0);
            gpu_time = magma_wtime();
            magma_zhesv( opts.uplo, N, opts.nrhs, h_A, lda, ipiv, h_X, ldb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zhesv returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) N, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) N, (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 0 ) {
                printf("     ---   \n");
            } else {
                error = get_residual( opts, opts.uplo, N, opts.nrhs, h_A, lda, ipiv, h_X, ldb, h_B, ldb );
                printf("   %8.2e   %s", error, (error < tol ? "ok" : "failed"));
                if (opts.lapack)
                    printf(" (lapack rel.res. = %8.2e)", error_lapack);
                printf("\n");
                status += ! (error < tol);
            }
            
            magma_free_cpu( ipiv );
            magma_free_pinned( h_X  );
            magma_free_pinned( h_B  );
            magma_free_pinned( h_A  );
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
