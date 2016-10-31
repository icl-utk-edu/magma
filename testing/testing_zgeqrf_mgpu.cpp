/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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
   -- Testing zgeqrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double           error, work[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
    magmaDoubleComplex_ptr d_lA[ MagmaMaxGPUs ];
    magma_int_t M, N, n2, lda, ldda, n_local, ngpu;
    magma_int_t info, min_mn, nb, lhwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1}, ISEED2[4];
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    opts.lapack |= (opts.check == 2);  // check (-c2) implies lapack (-l)
 
    int status = 0;
    double eps = lapackf77_dlamch("E");
    double tol = opts.tolerance * lapackf77_dlamch("E");

    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_create( dev, &queues[dev] );
    }
    
    printf("%% ngpu %lld\n", (long long) opts.ngpu );
    if ( opts.check == 1 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R-Q'A||_1 / (M*||A||_1) ||I-Q'Q||_1 / M\n");
        printf("%%===============================================================================================\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R||_F /(M*||A||_F)\n");
        printf("%%=========================================================================\n");
    }
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            nb     = magma_get_zgeqrf_nb( M, N );
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, magma_ceildiv(N,nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %lld GPUs\n", (long long) ngpu );
            }
            
            // query for workspace size
            lhwork = -1;
            lapackf77_zgeqrf( &M, &N, unused, &M, unused, tmp, &lhwork, &info );
            lhwork = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
            
            // Allocate host memory for the matrix
            TESTING_CHECK( magma_zmalloc_cpu( &tau,    min_mn ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_work, lhwork ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,    n2     ));
            
            // Allocate device memory
            for( int dev = 0; dev < ngpu; dev++ ) {
                n_local = ((N/nb)/ngpu)*nb;
                if (dev < (N/nb) % ngpu)
                    n_local += nb;
                else if (dev == (N/nb) % ngpu)
                    n_local += N % nb;
                magma_setdevice( dev );
                TESTING_CHECK( magma_zmalloc( &d_lA[dev], ldda*n_local ));
            }
            
            /* Initialize the matrix */
            for( int j=0; j < 4; j++ )
                ISEED2[j] = ISEED[j]; // save seeds
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                magmaDoubleComplex *tau2;
                TESTING_CHECK( magma_zmalloc_cpu( &tau2, min_mn ));
                cpu_time = magma_wtime();
                lapackf77_zgeqrf( &M, &N, h_A, &lda, tau2, h_work, &lhwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapack_zgeqrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                magma_free_cpu( tau2 );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix_1D_col_bcyclic( ngpu, M, N, nb, h_R, lda, d_lA, ldda, queues );

            gpu_time = magma_wtime();
            magma_zgeqrf2_mgpu( ngpu, M, N, d_lA, ldda, tau, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgeqrf2 returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_zgetmatrix_1D_col_bcyclic( ngpu, M, N, nb, d_lA, ldda, h_R, lda, queues );
            
            if ( opts.check == 1 && M >= N ) {
                /* =====================================================================
                   Check the result -- zqrt02 requires M >= N
                   =================================================================== */
                magma_int_t lwork = n2+N;
                magmaDoubleComplex *h_W1, *h_W2, *h_W3;
                double *h_RW, results[2];
            
                TESTING_CHECK( magma_zmalloc_cpu( &h_W1, n2    )); // Q
                TESTING_CHECK( magma_zmalloc_cpu( &h_W2, n2    )); // R
                TESTING_CHECK( magma_zmalloc_cpu( &h_W3, lwork )); // WORK
                TESTING_CHECK( magma_dmalloc_cpu( &h_RW, M ));  // RWORK
                lapackf77_zlarnv( &ione, ISEED2, &n2, h_A );
                lapackf77_zqrt02( &M, &N, &min_mn, h_A, h_R, h_W1, h_W2, &lda, tau, h_W3, &lwork,
                                  h_RW, results );
                results[0] *= eps;
                results[1] *= eps;

                if ( opts.lapack ) {
                    printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e                 %8.2e",
                           (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time, results[0], results[1] );
                } else {
                    printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)    %8.2e                 %8.2e",
                           (long long) M, (long long) N, gpu_perf, gpu_time, results[0], results[1] );
                }
                // todo also check results[1] < tol?
                printf("   %s\n", (results[0] < tol ? "ok" : "failed"));
                status += ! (results[0] < tol);

                magma_free_cpu( h_W1 );
                magma_free_cpu( h_W2 );
                magma_free_cpu( h_W3 );
                magma_free_cpu( h_RW );
            }
            else if ( opts.check == 2 ) {
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                error = lapackf77_zlange("f", &M, &N, h_A, &lda, work );
                blasf77_zaxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                error = lapackf77_zlange("f", &M, &N, h_R, &lda, work ) / (min_mn*error);
                
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                if ( opts.lapack ) {
                    printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   ---",
                           (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
                } else {
                    printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)     ---",
                           (long long) M, (long long) N, gpu_perf, gpu_time);
                }
                printf("%s\n", (opts.check != 0 ? "  (error check only for M >= N)" : ""));
            }
            
            magma_free_cpu( tau    );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_work );
            
            magma_free_pinned( h_R    );
            
            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_free( d_lA[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_destroy( queues[dev] );
    }
 
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
