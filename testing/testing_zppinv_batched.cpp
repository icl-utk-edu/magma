/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
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

#if defined(_OPENMP)
#include <omp.h>
#endif
#include "../control/magma_threadsetting.h"  // internal header

#define cond (N == 8 && batchCount == 1 && ibatch == 0)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpptri_batched
*/

int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *hA, *hAP, *hRP;
    magmaDoubleComplex *dAP;
    magma_int_t N, sizeA, sizeAP, sizeA_batch, sizeAP_batch, lda, info;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], error;
    int status = 0;
    magmaDoubleComplex **dAP_array = NULL;
    magma_int_t *dinfo_magma;
    magma_int_t *hinfo_magma;

    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    magma_queue_t queue = opts.queue;

    printf("%% BatchCount   N    CPU Gflop/s (ms)    GPU Gflop/s (ms)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = N;

            sizeA  = lda* N;
            sizeAP = (N+1) * N / 2;
            sizeA_batch  = sizeA  * batchCount;
            sizeAP_batch = sizeAP * batchCount;

            // FLOPS for inverse = POTRF + POTRI
            gflops = batchCount * ( FLOPS_ZPOTRF( N ) + FLOPS_ZPOTRI( N ) ) / 1e9;

            TESTING_CHECK( magma_imalloc_cpu( &hinfo_magma, batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &hA,  sizeA_batch  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hAP, sizeAP_batch ));
            TESTING_CHECK( magma_zmalloc_cpu( &hRP, sizeAP_batch ));

            TESTING_CHECK( magma_zmalloc( &dAP, sizeAP_batch ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));
            TESTING_CHECK( magma_malloc( (void**) &dAP_array, batchCount * sizeof(magmaDoubleComplex*) ));

            /* Initialize the matrix in column-major format*/
            lapackf77_zlarnv( &ione, ISEED, &sizeA_batch, hA );
            for (magma_int_t i=0; i < batchCount; i++) {
                magma_zmake_hpd( N, hA + i * lda * N, lda );
            }

            /* copy the matrix into packed format (hA -> hAP) */
            for (magma_int_t i=0; i < batchCount; i++) {
                magmaDoubleComplex *hAtmp  = hA  + i * ( N * lda );
                magmaDoubleComplex *hAPtmp = hAP + i * ( N * (N+1) / 2 );
                for(magma_int_t j=0; j < N; j++) {
                    magma_int_t length = N-j;
                    lapackf77_zlacpy( "F", &length, &ione, hAtmp + j*lda + j, &lda, hAPtmp, &length );
                    hAPtmp += length;
                }
            }

            // copy hAP to hRP
            size_t nelements = sizeAP * batchCount;
            memcpy( (void*)hRP, (const void*)hAP, nelements );

            // set matrix (packed format) cpu -> gpu
            for(magma_int_t i = 0; i < batchCount; i++) {
                magma_zsetvector( sizeAP, hAP + i * sizeAP, 1, dAP + i * sizeAP, 1, opts.queue );
            }

            for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                if(cond) {
                    magma_zprint(sizeAP, 1, hAP + ibatch*sizeAP, sizeAP);
                }
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_memset( dinfo_magma, 0, batchCount * sizeof(magma_int_t) );
            magma_zset_pointer( dAP_array, dAP,    1, 0, 0, sizeAP, batchCount, queue );

            if(opts.version  == 1) {
                // query workspace
                int64_t device_lwork[1] = {-1};
                void* device_work = NULL;
                info = magma_zppinv_batched( opts.uplo, N, NULL, device_work, device_lwork, NULL, batchCount, opts.queue);
                if(device_lwork[0] > 0) {
                    magma_malloc((void**)&device_work, device_lwork[0]);
                }

                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zppinv_batched( opts.uplo, N, dAP_array, device_work, device_lwork, dinfo_magma, batchCount, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                // free workspace, if any
                if( device_work != NULL) {
                    magma_free( device_work );
                }
            }
            else if( opts.version == 2 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zpptrf_batched( opts.uplo, N, dAP_array, dinfo_magma, batchCount, opts.queue );
                info = magma_zpptri_v2_batched_small( N, dAP_array, batchCount, dinfo_magma, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if( opts.version == 3 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zpptrf_batched( opts.uplo, N, dAP_array, dinfo_magma, batchCount, opts.queue );
                for(magma_int_t i = 0; i < batchCount; i++) {
                    magma_int_t locinfo = 0;
                    magma_zgetvector( sizeAP, dAP + i * sizeAP, 1, hRP + i * sizeAP, 1, opts.queue );
                    lapackf77_zpptri( lapack_uplo_const(opts.uplo), &N, hRP + i * sizeAP, &locinfo );
                    magma_zsetvector( sizeAP, hRP + i * sizeAP, 1, dAP + i * sizeAP, 1, opts.queue );
                }
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            gpu_perf = gflops / gpu_time;

            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, hinfo_magma, 1, opts.queue );
            for (int i=0; i < batchCount; i++) {
                if (hinfo_magma[i] != 0 ) {
                    printf("magma_zpptri_batched matrix %lld returned diag error %lld\n",
                            (long long) i, (long long) hinfo_magma[i] );
                    status = -1;
                }
            }

            if (info != 0) {
                printf("magma_zpptri_batched returned argument error %lld: %s.\n", (long long) info, magma_strerror( info ));
                status = -1;
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                magma_int_t nthreads = magma_get_lapack_numthreads();
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++) {
                    magma_int_t locinfo;
                    lapackf77_zpptrf( lapack_uplo_const(opts.uplo), &N, hAP + s * sizeAP, &locinfo );
                    lapackf77_zpptri( lapack_uplo_const(opts.uplo), &N, hAP + s * sizeAP, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zpptri matrix %lld returned error %lld: %s.\n",
                               (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;

                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                // get matrix (packed format) gpu -> cpu
                for(magma_int_t i = 0; i < batchCount; i++) {
                    magma_zgetvector( sizeAP, dAP + i * sizeAP, 1, hRP + i * sizeAP, 1, opts.queue );
                }

                for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                    if(cond) {
                        magma_zprint(sizeAP, 1, hRP + ibatch*sizeAP, sizeAP);
                        magma_zprint(sizeAP, 1, hAP + ibatch*sizeAP, sizeAP);
                    }
                }

                error = 0;
                for (int i=0; i < batchCount; i++) {
                    double Anorm, err;
                    blasf77_zaxpy(&sizeAP, &c_neg_one, hAP + i * sizeAP, &ione, hRP + i * sizeAP, &ione);
                    Anorm = lapackf77_zlange("f", &sizeAP, &ione, hAP + i * sizeAP, &sizeAP, work);
                    err   = lapackf77_zlange( "F", &sizeAP, &ione, hRP + i * sizeAP, &ione, work ) / Anorm;
                    if (std::isnan(err) || std::isinf(err)) {
                        error = err;
                        break;
                    }

                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;

                printf("%10lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) batchCount, (long long) N, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%10lld %5lld     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (long long) batchCount, (long long) N, gpu_perf, gpu_time*1000. );
            }

            magma_free_cpu( hinfo_magma );
            magma_free_cpu( hA  );
            magma_free_cpu( hAP );
            magma_free_cpu( hRP );
            magma_free( dAP );
            magma_free( dAP_array );
            magma_free( dinfo_magma );

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
