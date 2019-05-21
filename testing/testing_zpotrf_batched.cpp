/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>  // cudaMemset

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#endif
#include "../control/magma_threadsetting.h"  // internal header

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf_batched
*/

int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex *d_A;
    magma_int_t N, n2, lda, ldda, info;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], error;
    int status = 0;
    magmaDoubleComplex **d_A_array = NULL;
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
            n2  = lda* N  * batchCount;

            gflops = batchCount * FLOPS_ZPOTRF( N ) / 1e9;

            TESTING_CHECK( magma_imalloc_cpu( &hinfo_magma, batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2 ));
            TESTING_CHECK( magma_zmalloc( &d_A, ldda * N * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));
            
            TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(magmaDoubleComplex*) ));

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            for (int i=0; i < batchCount; i++)
            {
                magma_zmake_hpd( N, h_A + i * lda * N, lda ); // need modification
            }
            
            magma_int_t columns = N * batchCount;
            lapackf77_zlacpy( MagmaFullStr, &N, &(columns), h_A, &lda, h_R, &lda );

            magma_zsetmatrix( N, columns, h_A, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            cudaMemset( dinfo_magma, 0, batchCount * sizeof(magma_int_t) );

            magma_zset_pointer( d_A_array, d_A, ldda, 0, 0, ldda * N, batchCount, queue );
            gpu_time = magma_sync_wtime( opts.queue );
            info = magma_zpotrf_batched( opts.uplo, N, d_A_array, ldda, dinfo_magma, batchCount, queue);
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, hinfo_magma, 1, opts.queue );
            for (int i=0; i < batchCount; i++)
            {
                if (hinfo_magma[i] != 0 ) {
                    printf("magma_zpotrf_batched matrix %lld returned diag error %lld\n",
                            (long long) i, (long long) hinfo_magma[i] );
                    status = -1;
                }
            }
            if (info != 0) {
                //printf("magma_zpotrf_batched returned argument error %lld: %s.\n", (long long) info, magma_strerror( info ));
                status = -1;
            }                
            if (status == -1)
                goto cleanup;


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
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t locinfo;
                    lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A + s * lda * N, &lda, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zpotrf matrix %lld returned error %lld: %s.\n",
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
                magma_zgetmatrix( N, columns, d_A, ldda, h_R, lda, opts.queue );
                magma_int_t NN = lda*N;
                const char* uplo = lapack_uplo_const(opts.uplo);
                error = 0;
                for (int i=0; i < batchCount; i++)
                {
                    double Anorm, err;
                    blasf77_zaxpy(&NN, &c_neg_one, h_A + i * lda*N, &ione, h_R + i * lda*N, &ione);
                    Anorm = safe_lapackf77_zlanhe("f", uplo, &N, h_A + i * lda*N, &lda, work);
                    err   = safe_lapackf77_zlanhe("f", uplo, &N, h_R + i * lda*N, &lda, work)
                          / Anorm;
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
cleanup:
            magma_free_cpu( hinfo_magma );
            magma_free_cpu( h_A );
            magma_free_pinned( h_R );
            magma_free( d_A );
            magma_free( d_A_array );
            magma_free( dinfo_magma );
            if (status == -1)
                break;
            fflush( stdout );
        }
        if (status == -1)
            break;

        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
