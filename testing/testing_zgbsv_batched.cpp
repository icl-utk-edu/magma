/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Mark gates
   @author Ahmad Abdelfattah

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

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesv_batched
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time = 0;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t *dipiv, *dinfo_array;
    magma_int_t *ipiv, *cpu_info;
    magma_int_t N, Nband, KL, KU, KV, nrhs, lda, ldb, ldda, lddb, info = 0, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dB_array = NULL;
    magma_int_t     **dipiv_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    nrhs       = opts.nrhs;
    KL         = opts.kl;
    KU         = opts.ku;
    KV         = KL + KU;
    batchCount = opts.batchcount;

    printf("%% ## INFO ##: Gflop/s calculation is not available\n");
    printf("%% Lower bandwidth (KL) = %lld\n", (long long)KL);
    printf("%% Upper bandwidth (KU) = %lld\n", (long long)KU);
    printf("%% BatchCount   N  NRHS   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            Nband  = KL + 1 + KV; // need extra KL superdiagonals for the upper factor
            lda    = Nband;
            ldb    = N;
            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( ldb, opts.align );
            gflops = 0.;  // TODO: gflop formula for gbsv?

            sizeA = lda*N*batchCount;
            sizeB = ldb*nrhs*batchCount;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B, sizeB ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, sizeB ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, batchCount*N ));
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));

            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N*batchCount    ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs*batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv, N * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );

            // random initialization of h_A seems to produce
            // some matrices that are singular, the additive statements below
            // seem to avoid that
            #pragma omp parallel for schedule(dynamic)
            for(int s = 0; s < batchCount; s++) {
                magmaDoubleComplex* hA = h_A + s*lda*N;
                for(int j = 0; j < lda*N; j++) {
                    MAGMA_Z_REAL( hA[j] ) += 20.;
                    #if defined(PRECISION_c) || defined(PRECISION_z)
                    MAGMA_Z_IMAG( hA[j] ) += 20.;
                    #endif
                }
            }

            magma_zsetmatrix( Nband, N*batchCount,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N,     nrhs*batchCount, h_B, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N,    batchCount, opts.queue );
            magma_zset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, opts.queue );
            magma_iset_pointer( dipiv_array, dipiv, 1, 0, 0, N, batchCount, opts.queue );
            if(opts.version == 1) {
                // synchronous api with ptr array
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zgbsv_batched(
                        N, KL, KU, nrhs,
                        dA_array, ldda, dipiv_array,
                        dB_array, lddb, dinfo_array,
                        batchCount, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if(opts.version == 2) {
                // synchronous api with stride
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zgbsv_batched_strided(
                        N, KL, KU, nrhs,
                        d_A, ldda, ldda*N,
                        dipiv, N,
                        d_B, lddb, lddb*nrhs,
                        dinfo_array, batchCount, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if(opts.version == 3) {
                // async api with ptr array
                // query workspace
                magma_int_t lwork[1] = {-1};
                magma_zgbsv_batched_work(
                        N, KL, KU, nrhs,
                        NULL, ldda, NULL,
                        NULL, lddb,
                        NULL, NULL, lwork, batchCount, opts.queue);

                // allocate workspace
                void* device_work = NULL;
                TESTING_CHECK( magma_malloc(&device_work, lwork[0]) );

                // time the async interface only
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zgbsv_batched_work(
                        N, KL, KU, nrhs,
                        dA_array, ldda, dipiv_array,
                        dB_array, lddb,
                        dinfo_array, device_work, lwork, batchCount, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                // free workspace
                magma_free( device_work );
            }
            else if(opts.version == 4) {
                // async api with stride
                // query workspace
                magma_int_t lwork[1] = {-1};
                magma_zgbsv_batched_strided_work(
                    N, KL, KU, nrhs,
                    NULL, ldda, ldda*N, NULL, N,
                    NULL, lddb, lddb*nrhs, NULL, NULL, lwork,
                    batchCount, opts.queue);

                // allocate workspace
                void* device_work = NULL;
                TESTING_CHECK( magma_malloc(&device_work, lwork[0]) );

                // time the async interface only
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zgbsv_batched_strided_work(
                        N, KL, KU, nrhs,
                        d_A, ldda, ldda*N, dipiv, N,
                        d_B, lddb, lddb*nrhs,
                        dinfo_array, device_work, lwork,
                        batchCount, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                // free workspace
                magma_free( device_work );
            }
            gpu_perf = gflops / gpu_time;

            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, opts.queue );
            if (info != 0) {
                printf("magma_zgbsv_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }
            else {
                for (int i=0; i < batchCount; i++) {
                    if (cpu_info[i] != 0 ) {
                        printf("magma_zgbsv_batched matrix %lld returned internal error %lld\n",
                                (long long) i, (long long) cpu_info[i] );
                    }
                }
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, opts.queue );
            error = 0;
            for (magma_int_t s=0; s < batchCount; s++) {
                magmaDoubleComplex* hA = h_A + s * lda * N + KL;
                magmaDoubleComplex* hX = h_X + s * ldb * nrhs;
                magmaDoubleComplex* hB = h_B + s * ldb * nrhs;

                Anorm = lapackf77_zlangb("I", &N, &KL, &KU, hA, &lda, work);
                Xnorm = lapackf77_zlange("I", &N, &nrhs, hX, &ldb, work);

                for(magma_int_t j = 0; j < nrhs; j++) {
                    blasf77_zgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
                                   &c_one, hA           , &lda,
                                           hX  + j * ldb, &ione,
                               &c_neg_one, hB  + j * ldb, &ione);
                }

                Rnorm = lapackf77_zlange("I", &N, &nrhs, hB, &ldb, work);

                double err = Rnorm/(N*Anorm*Xnorm);
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
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t locinfo;
                    lapackf77_zgbsv(
                        &N, &KL, &KU, &nrhs,
                        h_A  + s*lda*N,    &lda, ipiv + s*N,
                        h_B  + s*ldb*nrhs, &ldb, &locinfo );

                    if (locinfo != 0) {
                        printf("lapackf77_zgesv matrix %lld returned error %lld: %s.\n",
                                (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;

                printf( "%10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10lld %5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        gpu_perf, 1000.*gpu_time,
                        error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );
            magma_free_cpu( cpu_info );

            magma_free( d_A );
            magma_free( d_B );

            magma_free( dipiv );
            magma_free( dinfo_array );

            magma_free( dA_array );
            magma_free( dB_array );
            magma_free( dipiv_array );
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
