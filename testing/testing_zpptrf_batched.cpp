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

#define cond (N == 8 && batchCount == 4 && ibatch == 3)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpptrf_batched
*/

int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *hA, *hAP, *hRP;
    magmaDoubleComplex *dA, *dAP;
    magma_int_t N, sizeA, sizeAP, sizeA_batch, sizeAP_batch, lda, ldda, info;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], error;
    int status = 0;
    magmaDoubleComplex **dA_array = NULL, **dAP_array = NULL;
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
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default

            sizeA  = lda* N;
            sizeAP = (N+1) * N / 2;
            sizeA_batch  = sizeA  * batchCount;
            sizeAP_batch = sizeAP * batchCount;

            // use the same flop count as POTRF
            gflops = batchCount * FLOPS_ZPOTRF( N ) / 1e9;

            TESTING_CHECK( magma_imalloc_cpu( &hinfo_magma, batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &hA,  sizeA_batch  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hAP, sizeAP_batch ));
            TESTING_CHECK( magma_zmalloc_cpu( &hRP, sizeAP_batch ));

            TESTING_CHECK( magma_zmalloc( &dA,  ldda * N * batchCount ));
            TESTING_CHECK( magma_zmalloc( &dAP, sizeAP_batch ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,  batchCount * sizeof(magmaDoubleComplex*) ));
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

            // set matrix (col major) cpu -> gpu
            magma_int_t NN = batchCount * N;
            magma_zsetmatrix(N, NN, hA, lda, dA, ldda, queue);

            // set matrix (packed format) cpu -> gpu
            for(magma_int_t i = 0; i < batchCount; i++) {
                magma_zsetvector( sizeAP, hAP + i * sizeAP, 1, dAP + i * sizeAP, 1, opts.queue );
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_memset( dinfo_magma, 0, batchCount * sizeof(magma_int_t) );

            // set pointer array for packed matrices
            magma_zset_pointer( dAP_array, dAP,    1, 0, 0, sizeAP, batchCount, queue );
            magma_zset_pointer( dA_array,   dA, ldda, 0, 0, ldda*N, batchCount, queue );


            if( opts.version == 1 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zpptrf_batched( opts.uplo, N, dAP_array, dinfo_magma, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if( opts.version == 2 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zpptrf_batched_small( opts.uplo, N, dAP_array, dinfo_magma, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if( opts.version == 3 ) {
                // slow ref. impl.
                gpu_time = magma_sync_wtime( opts.queue );
                // copy dAP to dA
                for(magma_int_t i = 0; i < batchCount; i++) {
                    magmaDoubleComplex *dAtmp  = dA  + i * ( N * ldda );
                    magmaDoubleComplex *dAPtmp = dAP + i * ( N * (N+1) / 2 );
                    for(magma_int_t j = 0; j < N; j++) {
                        magma_int_t length = N-j;
                        magma_zcopyvector(length, dAPtmp, 1, dAtmp + j*ldda + j, 1, opts.queue );
                        dAPtmp += length;
                    }
                }

                for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                    if(cond) {
                        magma_zprint_gpu(     N, N, dA  + ibatch*ldda*N,   ldda, opts.queue);
                        magma_zprint_gpu(sizeAP, 1, dAP + ibatch*sizeAP, sizeAP, opts.queue);
                    }
                }

                // call magma batch potrf (col-major)
                info = magma_zpotrf_batched( opts.uplo, N, dA_array, ldda, dinfo_magma, batchCount, queue);

                for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                    if(cond) {
                        magma_zprint_gpu(N, N, dA+ibatch*ldda*N, ldda, opts.queue);
                    }
                }

                // copy results back from dA --> dAP
                for(magma_int_t i = 0; i < batchCount; i++) {
                    magmaDoubleComplex *dAtmp  = dA  + i * ( N * ldda );
                    magmaDoubleComplex *dAPtmp = dAP + i * sizeAP;
                    for(magma_int_t j = 0; j < N; j++) {
                        magma_int_t length = N-j;
                        magma_zcopyvector(length, dAtmp + j*ldda + j, 1, dAPtmp, 1, opts.queue );
                        dAPtmp += length;
                    }
                }
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                    if(cond) {
                        magma_zprint_gpu(sizeAP, 1, dAP + ibatch*sizeAP, sizeAP, opts.queue);
                    }
                }

            }
            gpu_perf = gflops / gpu_time;

            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, hinfo_magma, 1, opts.queue );
            for (int i=0; i < batchCount; i++) {
                if (hinfo_magma[i] != 0 ) {
                    printf("magma_zpptrf_batched matrix %lld returned diag error %lld\n",
                            (long long) i, (long long) hinfo_magma[i] );
                    status = -1;
                }
            }

            if (info != 0) {
                printf("magma_zpptrf_batched returned argument error %lld: %s.\n", (long long) info, magma_strerror( info ));
                status = -1;
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++) {
                    magma_int_t locinfo;
                    lapackf77_zpptrf( lapack_uplo_const(opts.uplo), &N, hAP + s * sizeAP, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zpptrf matrix %lld returned error %lld: %s.\n",
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
                //magma_zgetmatrix( N, columns, dAP, ldda, hRP, lda, opts.queue );
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
            magma_free( dA  );
            magma_free( dAP );
            magma_free( dA_array  );
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
