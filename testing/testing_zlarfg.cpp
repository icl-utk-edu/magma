/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_x, *h_x2, *h_tau, *h_tau2;
    magmaDoubleComplex_ptr d_x, d_tau;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double      error, error2, work[1];
    magma_int_t N, nb, lda, ldda, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    // does larfg on nb columns, one after another
    nb = (opts.nb > 0 ? opts.nb : 64);
    
    printf("%%   N    nb    CPU GFLop/s (ms)    GPU Gflop/s (ms)   error      tau error\n");
    printf("%%=========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda  = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZLARFG( N ) / 1e9 * nb;
    
            TESTING_CHECK( magma_zmalloc_cpu( &h_x,    N*nb ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_x2,   N*nb ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_tau,  nb   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_tau2, nb   ));
        
            TESTING_CHECK( magma_zmalloc( &d_x,   ldda*nb ));
            TESTING_CHECK( magma_zmalloc( &d_tau, nb      ));
            
            /* Initialize the vectors */
            size = N*nb;
            lapackf77_zlarnv( &ione, ISEED, &size, h_x );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( N, nb, h_x, N, d_x, ldda, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            for( int j = 0; j < nb; ++j ) {
                magmablas_zlarfg( N, &d_x[0+j*ldda], &d_x[1+j*ldda], ione, &d_tau[j], opts.queue );
            }
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            magma_zgetmatrix( N, nb, d_x, ldda, h_x2, N, opts.queue );
            magma_zgetvector( nb, d_tau, 1, h_tau2, 1, opts.queue );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int j = 0; j < nb; ++j ) {
                lapackf77_zlarfg( &N, &h_x[0+j*lda], &h_x[1+j*lda], &ione, &h_tau[j] );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Error Computation and Performance Comparison
               =================================================================== */
            blasf77_zaxpy( &size, &c_neg_one, h_x, &ione, h_x2, &ione );
            error = lapackf77_zlange( "F", &N, &nb, h_x2, &N, work )
                  / lapackf77_zlange( "F", &N, &nb, h_x,  &N, work );
            
            // tau can be 0
            blasf77_zaxpy( &nb, &c_neg_one, h_tau, &ione, h_tau2, &ione );
            error2 = lapackf77_zlange( "F", &nb, &ione, h_tau,  &nb, work );
            if ( error2 != 0 ) {
                error2 = lapackf77_zlange( "F", &nb, &ione, h_tau2, &nb, work ) / error2;
            }

            printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                   (long long) N, (long long) nb, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                   error, error2,
                   (error < tol && error2 < tol ? "ok" : "failed") );
            status += ! (error < tol && error2 < tol);
            
            magma_free_cpu( h_x    );
            magma_free_cpu( h_x2   );
            magma_free_cpu( h_tau  );
            magma_free_cpu( h_tau2 );
        
            magma_free( d_x   );
            magma_free( d_tau );
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
