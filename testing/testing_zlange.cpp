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
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlange
*/
int main( int argc, char** argv)
{
    #define h_A(i_, j_) (h_A + (i_) + (j_)*lda)
    
    #ifdef HAVE_clBLAS
    #define d_A(i_, j_)  d_A, ((i_) + (j_)*ldda)
    #else
    #define d_A(i_, j_) (d_A + (i_) + (j_)*ldda)
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A;
    double *h_work;
    magmaDoubleComplex_ptr d_A;
    magmaDouble_ptr d_work;
    magma_int_t i, j, M, N, n2, lda, ldda, lwork;
    magma_int_t idist    = 3;  // normal distribution (otherwise max norm is always ~ 1)
    magma_int_t ISEED[4] = {0,0,0,1};
    double      error, norm_magma, norm_lapack, normalize, tol;
    int status = 0;
    magma_int_t lapack_nan_fail = 0;
    magma_int_t lapack_inf_fail = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double eps = lapackf77_dlamch("E");
    
    magma_norm_t norm[] = { MagmaMaxNorm, MagmaOneNorm, MagmaInfNorm, MagmaFrobeniusNorm };
    
    printf("%%   M     N   norm   CPU GByte/s (ms)    GPU GByte/s (ms)        error               nan      inf\n");
    printf("%%================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int inorm = 0; inorm < 4; ++inorm ) {  /* < 4 for Frobenius */
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M   = opts.msize[itest];
            N   = opts.nsize[itest];
            lda = M;
            n2  = lda*N;
            ldda = magma_roundup( M, opts.align );
            if ( norm[inorm] == MagmaOneNorm || norm[inorm] == MagmaFrobeniusNorm )
                lwork = N;
            else
                lwork = M;
            // read whole matrix
            gbytes = M*N*sizeof(magmaDoubleComplex) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    n2 ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_work, M ));
            
            TESTING_CHECK( magma_zmalloc( &d_A,    ldda*N ));
            TESTING_CHECK( magma_dmalloc( &d_work, lwork ));
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &idist, ISEED, &n2, h_A );
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            norm_magma = magmablas_zlange( norm[inorm], M, N, d_A, ldda, d_work, lwork, opts.queue );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (norm_magma == -1) {
                printf( "%5lld   %4c   skipped because %s norm isn't supported\n",
                        (long long) N, lapacke_norm_const( norm[inorm] ), lapack_norm_const( norm[inorm] ));
                goto cleanup;
            }
            else if (norm_magma < 0) {
                printf("magmablas_zlange returned error %f: %s.\n",
                       norm_magma, magma_strerror( magma_int_t(norm_magma) ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            norm_lapack = lapackf77_zlange( lapack_norm_const(norm[inorm]), &M, &N, h_A, &lda, h_work );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (norm_lapack < 0) {
                printf("lapackf77_zlange returned error %f: %s.\n",
                       norm_lapack, magma_strerror( magma_int_t(norm_lapack) ));
            }
            
            /* =====================================================================
               Check the result compared to LAPACK
               One, Inf, Fro errors normalized by sqrt of # terms in summation.
               =================================================================== */
            normalize = 1;
            tol = 3*eps;
            if ( norm[inorm] == MagmaMaxNorm ) {
                // max-norm depends on only one element, so for Real precisions,
                // MAGMA and LAPACK should exactly agree (tol = 0),
                // while Complex precisions incur roundoff in cuCabs.
                #ifdef REAL
                tol = 0;
                #endif
            }
            else if ( norm[inorm] == MagmaOneNorm ) {
                normalize = sqrt( (double)M );
            }
            else if ( norm[inorm] == MagmaInfNorm ) {
                normalize = sqrt( (double)N );
            }
            else if ( norm[inorm] == MagmaFrobeniusNorm ) {
                normalize = sqrt( (double)M*N );
            }
            error = fabs( norm_magma - norm_lapack )
                  / (norm_lapack * normalize);
            bool okay; okay = (error <= tol);
            status += ! okay;
            
            /* ====================================================================
               Check for NAN and INF propagation
               =================================================================== */
            i = rand() % M;
            j = rand() % N;
            *h_A(i,j) = MAGMA_Z_NAN;
            magma_zsetvector( 1, h_A(i,j), 1, d_A(i,j), 1, opts.queue );
            norm_magma  = magmablas_zlange( norm[inorm], M, N, d_A, ldda, d_work, lwork, opts.queue );
            norm_lapack = lapackf77_zlange( lapack_norm_const( norm[inorm] ),
                                            &M, &N, h_A, &lda, h_work );
            bool nan_okay;    nan_okay    = std::isnan(norm_magma);
            bool la_nan_okay; la_nan_okay = std::isnan(norm_lapack);
            lapack_nan_fail += ! la_nan_okay;
            status          += !    nan_okay;
            
            *h_A(i,j) = MAGMA_Z_INF;
            magma_zsetvector( 1, h_A(i,j), 1, d_A(i,j), 1, opts.queue );
            norm_magma  = magmablas_zlange( norm[inorm], M, N, d_A, ldda, d_work, lwork, opts.queue );
            norm_lapack = lapackf77_zlange( lapack_norm_const( norm[inorm] ),
                                            &M, &N, h_A, &lda, h_work );
            bool inf_okay;    inf_okay    = std::isinf(norm_magma);
            bool la_inf_okay; la_inf_okay = std::isinf(norm_lapack);
            lapack_inf_fail += ! la_inf_okay;
            status          += !    inf_okay;
            
            printf("%5lld %5lld   %4c   %7.2f (%7.2f)   %7.2f (%7.2f)   %#9.3g   %-6s   %6s%1s  %6s%1s\n",
                   (long long) M, (long long) N,
                   lapacke_norm_const( norm[inorm] ),
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error,
                   (okay     ? "ok" : "failed"),
                   (nan_okay ? "ok" : "failed"), (la_nan_okay ? " " : "*"),
                   (inf_okay ? "ok" : "failed"), (la_inf_okay ? " " : "*"));
            
        cleanup:
            magma_free_cpu( h_A    );
            magma_free_cpu( h_work );
            
            magma_free( d_A    );
            magma_free( d_work );
            fflush( stdout );
        } // end iter
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      } // end inorm
      printf( "\n" );
    }
    
    if ( lapack_nan_fail ) {
        printf( "* Warning: LAPACK did not pass NAN propagation test; upgrade to LAPACK version >= 3.4.2 (Sep. 2012)\n" );
    }
    if ( lapack_inf_fail ) {
        printf( "* Warning: LAPACK did not pass INF propagation test\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
