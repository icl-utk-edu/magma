/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Mark gates
   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
 */
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#define cond (N == 8 && batchCount == 1 && ibatch == 0)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zppsv_batched
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *hA, *hAP, *hB, *hX;
    magmaDoubleComplex_ptr dA, dAP, dB;
    magma_int_t *cpu_info;
    magma_int_t *dinfo_array;
    magma_int_t N, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeAP, sizeA_batch, sizeAP_batch, sizeB, sizeB_batch;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaDoubleComplex **dA_array = NULL, **dAP_array = NULL;
    magmaDoubleComplex **dB_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    magma_queue_t queue = opts.queue;

    nrhs = opts.nrhs;
    batchCount = opts.batchcount;

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% BatchCount   N  NRHS   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%==========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb   = ldda;

            // use the same flop count as POSV
            gflops = ( FLOPS_ZPOTRF( N) + FLOPS_ZPOTRS( N, nrhs ) ) / 1e9 * batchCount;

            sizeA        = lda*N;
            sizeB        = ldb*nrhs;
            sizeAP       = (N+1) * N / 2;
            sizeA_batch  = sizeA  * batchCount;
            sizeAP_batch = sizeAP * batchCount;
            sizeB_batch  = sizeB  * batchCount;

            TESTING_CHECK( magma_zmalloc_cpu( &hA,  sizeA_batch  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hAP, sizeAP_batch ));
            TESTING_CHECK( magma_zmalloc_cpu( &hB,  sizeB_batch  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hX,  sizeB_batch  ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));


            TESTING_CHECK( magma_zmalloc( &dA,  ldda*N*batchCount    ));
            TESTING_CHECK( magma_zmalloc( &dB,  lddb*nrhs*batchCount ));
            TESTING_CHECK( magma_zmalloc( &dAP, sizeAP_batch         ));

            TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,  batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dAP_array, batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array,  batchCount * sizeof(magmaDoubleComplex*) ));

            /* Initialize the matrix in column-major format*/
            lapackf77_zlarnv( &ione, ISEED, &sizeA_batch, hA );
            lapackf77_zlarnv( &ione, ISEED, &sizeB_batch, hB );
            for (int i=0; i < batchCount; i++) {
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

            // set matrix (col major) cpu -> gpu
            magma_zsetmatrix( N, N*batchCount,    hA, lda, dA, ldda, opts.queue );
            magma_zsetmatrix( N, nrhs*batchCount, hB, ldb, dB, lddb, opts.queue );

            // set matrix (packed format) cpu -> gpu
            for(magma_int_t i = 0; i < batchCount; i++) {
                magma_zsetvector( sizeAP, hAP + i * sizeAP, 1, dAP + i * sizeAP, 1, opts.queue );
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( dA_array,  dA,  ldda, 0, 0, ldda*N,    batchCount, queue );
            magma_zset_pointer( dAP_array, dAP,    1, 0, 0, sizeAP,    batchCount, queue );
            magma_zset_pointer( dB_array,  dB,  lddb, 0, 0, lddb*nrhs, batchCount, queue );

            for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                if(cond) {
                    magma_zprint_gpu(N,    N, dA+ibatch*ldda*N,    ldda, opts.queue);
                    magma_zprint_gpu(N, nrhs, dB+ibatch*lddb*nrhs, lddb, opts.queue);
                }
            }

            if( opts.version == 1 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zppsv_batched(opts.uplo, N, nrhs, dAP_array, dB_array, lddb, dinfo_array, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if( opts.version == 2 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zpptrf_batched_small( opts.uplo, N, dAP_array, dinfo_array, batchCount, opts.queue );

                for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                    if(cond) {
                        magma_zprint_gpu(sizeAP, 1, dAP+ibatch*sizeAP, sizeAP, opts.queue);
                        magma_zprint_gpu(N, nrhs, dB+ibatch*lddb*nrhs, lddb, opts.queue);
                    }
                }

                magma_zpptrs_batched_small(N, nrhs, dAP_array, dB_array, lddb, batchCount, opts.queue );
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

                // solve in column major
                info = magma_zposv_batched(opts.uplo, N, nrhs, dA_array, ldda, dB_array, lddb, dinfo_array, batchCount, queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            gpu_perf = gflops / gpu_time;

            // check correctness of results throught "dinfo_array" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, opts.queue );
            for (int i=0; i < batchCount; i++) {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zppsv_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                }
            }

            if (info != 0) {
                printf("magma_zppsv_batched returned argument error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs*batchCount, dB, lddb, hX, ldb, opts.queue );

            for(magma_int_t ibatch = 0; ibatch < batchCount; ibatch++) {
                if(cond) {
                    magma_zprint_gpu(N, nrhs, dB+ibatch*lddb*nrhs, lddb, opts.queue);
                }
            }

            error = 0;
            for (magma_int_t s=0; s < batchCount; s++) {
                Anorm = lapackf77_zlange("I", &N, &N,    hA + s * lda * N, &lda, work);
                Xnorm = lapackf77_zlange("I", &N, &nrhs, hX + s * ldb * nrhs, &ldb, work);

                blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     hA + s * lda * N, &lda,
                                       hX + s * ldb * nrhs, &ldb,
                           &c_neg_one, hB + s * ldb * nrhs, &ldb);

                Rnorm = lapackf77_zlange("I", &N, &nrhs, hB + s * ldb * nrhs, &ldb, work);
                double err = Rnorm/(N*Anorm*Xnorm);

                //printf("err[%3d] = %.4e\n", s, err);
                if (std::isnan(err) || std::isinf(err)) {
                    error = err;
                    break;
                }
                error = max( err, error );
            }
            bool okay = (error < tol);
            status += ! okay;

            /* ====================================================================
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
                    lapackf77_zppsv( lapack_uplo_const(opts.uplo), &N, &nrhs, hAP + s * sizeAP, hB + s * ldb * nrhs, &ldb, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zppsv matrix %lld returned error %lld: %s.\n",
                               (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;

                printf( "%10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10lld %5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs, gpu_perf, gpu_time*1000.,
                        error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( hA  );
            magma_free_cpu( hAP );
            magma_free_cpu( hB  );
            magma_free_cpu( hX  );
            magma_free_cpu( work );
            magma_free_cpu( cpu_info );

            magma_free( dA  );
            magma_free( dAP );
            magma_free( dB  );

            magma_free( dinfo_array );

            magma_free( dA_array  );
            magma_free( dAP_array );
            magma_free( dB_array  );

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
