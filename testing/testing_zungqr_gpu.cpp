/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Stan Tomov
       @author Mathieu Faverge
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zungqr_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double          Anorm, error, work[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *hA, *hR, *tau, *h_work;
    magmaDoubleComplex_ptr dA, dT;
    magma_int_t m, n, k;
    magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
    magma_int_t ione     = 1;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    printf("%%   m     n     k   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R|| / ||A||\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            k = opts.ksize[itest];
            if ( m < n || n < k ) {
                printf( "%5lld %5lld %5lld   skipping because m < n or n < k\n", (long long) m, (long long) n, (long long) k );
                continue;
            }
            
            lda  = m;
            ldda = magma_roundup( m, opts.align );  // multiple of 32 by default
            n2 = lda*n;
            min_mn = min(m, n);
            nb = magma_get_zgeqrf_nb( m, n );
            lwork  = n*nb;
            gflops = FLOPS_ZUNGQR( m, n, k ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &hA,     lda*n  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hR,     lda*n  ));
            TESTING_CHECK( magma_zmalloc_cpu( &tau,    min_mn ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_work, lwork  ));
            
            TESTING_CHECK( magma_zmalloc( &dA,     ldda*n ));
            TESTING_CHECK( magma_zmalloc( &dT,     ( 2*min_mn + magma_roundup( n, 32 ) )*nb ));
            
            magma_generate_matrix( opts, m, n, hA, lda );
            lapackf77_zlacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            
            Anorm = lapackf77_zlange("f", &m, &n, hA, &lda, work );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // first, get QR factors in both hA and dA
            // okay that magma_zgeqrf_gpu has special structure for R; R isn't used here.
            magma_zsetmatrix(  m, n, hA, lda, dA, ldda, opts.queue );
            magma_zgeqrf_gpu( m, n, dA, ldda, tau, dT, &info );
            if (info != 0) {
                printf("magma_zgeqrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            magma_zgetmatrix( m, n, dA, ldda, hA, lda, opts.queue );
            
            gpu_time = magma_wtime();
            magma_zungqr_gpu( m, n, k, dA, ldda, tau, dT, nb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zungqr_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            // Get dA back to the CPU to compare with the CPU result.
            magma_zgetmatrix( m, n, dA, ldda, hR, lda, opts.queue );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zungqr( &m, &n, &k, hA, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zungqr returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
                blasf77_zaxpy( &n2, &c_neg_one, hA, &ione, hR, &ione );
                error = lapackf77_zlange("f", &m, &n, hR, &lda, work) / Anorm;
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%5lld %5lld %5lld   %7.1f (%7.2f)   %7.1f (%7.2f)   %8.2e   %s\n",
                       (long long) m, (long long) n, (long long) k,
                       cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld %5lld %5lld     ---   (  ---  )   %7.1f (%7.2f)     ---  \n",
                       (long long) m, (long long) n, (long long) k,
                       gpu_perf, gpu_time );
            }
            
            magma_free_cpu( hA     );
            magma_free_cpu( hR  );
            magma_free_cpu( tau );
            magma_free_cpu( h_work );
            
            magma_free( dA );
            magma_free( dT );
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
