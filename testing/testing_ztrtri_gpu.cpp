/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
  
       @precisions normal z -> c d s
       
       @author Mark Gates
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
   -- Testing ztrtri
*/
int main( int argc, char** argv)
{
    #define h_A(i_, j_) (h_A + (i_) + (j_)*lda)

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_A;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t N, n2, lda, ldda, info;
    magma_int_t *ipiv;
    magma_int_t ione     = 1;
    double      work[1], error, norm;
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default; makes triangles nicely conditioned
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R||_F / ||A||_F\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZTRTRI( N ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2 ));
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N ));
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zgetrf( &N, &N, h_A, &lda, ipiv, &info );
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < j; ++i) {
                    *h_A(i,j) = *h_A(j,i);
                }
            }
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            
            // check for exact singularity
            //magma_zgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
            //h_R[ 10 + 10*lda ] = MAGMA_Z_MAKE( 0.0, 0.0 );
            //magma_zsetmatrix( N, N, h_R, lda, d_A, ldda, opts.queue );
            
            gpu_time = magma_wtime();
            magma_ztrtri_gpu( opts.uplo, opts.diag, N, d_A, ldda, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_ztrtri_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_ztrtri( lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_ztrtri returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_zgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
                if ( opts.verbose ) {
                    printf( "A=" );  magma_zprint( N, N, h_A, lda );
                    printf( "R=" );  magma_zprint( N, N, h_R, lda );
                }
                norm  = lapackf77_zlantr("f", lapack_uplo_const(opts.uplo), MagmaNonUnitStr, &N, &N, h_A, &lda, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_zlantr("f", lapack_uplo_const(opts.uplo), MagmaNonUnitStr, &N, &N, h_R, &lda, work) / norm;
                if ( opts.verbose ) {
                    printf( "diff=" );  magma_zprint( N, N, h_R, lda );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (okay ? "ok" : "failed") );
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (long long) N, gpu_perf, gpu_time );
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( ipiv );
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
