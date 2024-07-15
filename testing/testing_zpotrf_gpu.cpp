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
   -- Testing zpotrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    // constants
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;

    // locals
    real_Double_t   gflops, gpu_perf=0, gpu_time=0, cpu_perf=0, cpu_time=0;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_A;
    magma_int_t N, n2, lda, ldda, info;
    double      Anorm, error, work[1], *sigma;
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)

    double tol = opts.tolerance * lapackf77_dlamch("E");

    // for expert API testing
    magma_device_t cdev;
    magma_queue_t queues[2];
    magma_event_t events[2];
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% N     CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = max(1, N);
            n2  = lda*N;
            ldda = max(1, magma_roundup( N, opts.align ));  // multiple of 32 by default
            gflops = FLOPS_ZPOTRF( N ) / 1e9;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( magma_dmalloc_cpu( &sigma, N ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2 ));
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N ));

            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda, sigma );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            if(opts.version == 1){
                gpu_time = magma_wtime();
                magma_zpotrf_gpu( opts.uplo, N, d_A, ldda, &info );
                gpu_time = magma_wtime() - gpu_time;
            }
            else if(opts.version == 2){
                gpu_time = magma_wtime();
                magma_zpotrf_native(opts.uplo, N, d_A, ldda, &info );
                gpu_time = magma_wtime() - gpu_time;
            }
            else if(opts.version == 3 || opts.version == 4) {
                // expert interface
                magma_mode_t mode = (opts.version == 3) ? MagmaHybrid : MagmaNative;
                magma_int_t nb    = magma_get_zpotrf_nb( N );
                magma_int_t recnb = 128;

                // query workspace
                void *hwork = NULL, *dwork=NULL;
                magma_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
                magma_zpotrf_expert_gpu_work(
                    opts.uplo, N, NULL, ldda, &info,
                    mode, nb, recnb,
                    NULL, lhwork, NULL, ldwork,
                    events, queues );

                // alloc workspace
                if( lhwork[0] > 0 ) {
                    magma_malloc_pinned( (void**)&hwork, lhwork[0] );
                }

                if( ldwork[0] > 0 ) {
                    magma_malloc( (void**)&dwork, ldwork[0] );
                }

                // time actual call only
                gpu_time = magma_wtime();
                magma_zpotrf_expert_gpu_work(
                    opts.uplo, N, d_A, ldda, &info,
                    mode, nb, recnb,
                    hwork, lhwork, dwork, ldwork,
                    events, queues );
                magma_queue_sync( queues[0] );
                magma_queue_sync( queues[1] );
                gpu_time = magma_wtime() - gpu_time;

                // free workspace
                if( hwork != NULL) {
                    magma_free_pinned( hwork );
                }

                if( dwork != NULL ) {
                    magma_free( dwork );
                }
            }

            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zpotrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zpotrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }

                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_zgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                #ifndef MAGMA_HAVE_HIP
                Anorm = lapackf77_zlange("f", &N, &N, h_A, &lda, work);
                error = lapackf77_zlange("f", &N, &N, h_R, &lda, work) / Anorm;
                #else
                // TODO: use zlange when the herk/syrk implementations are standardized.
                // For HIP, the current herk/syrk routines overwrite the entire diagonal
                // blocks of the matrix, so using zlange causes the error check to fail
                Anorm = safe_lapackf77_zlanhe( "f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_zlanhe( "f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work ) / Anorm;
                #endif

                if (N == 0)
                    error = 0.0;

                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (long long) N, gpu_perf, gpu_time );
            }
            magma_free_cpu( h_A );
            magma_free_cpu( sigma );
            magma_free_pinned( h_R );
            magma_free( d_A );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
