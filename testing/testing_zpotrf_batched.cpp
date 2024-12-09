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

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#ifdef MAGMA_HAVE_CUDA
#include <cusolverDn.h>
#else
#include<rocblas/rocblas.h>
#include<rocsolver/rocsolver.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif
#include "../control/magma_threadsetting.h"  // internal header

#define PRECISION_z

#ifdef MAGMA_HAVE_CUDA
#define devsolver_handle_t  cusolverDnHandle_t
#define devsolver_create    cusolverDnCreate
#define devsolver_setstream cusolverDnSetStream
#define devsolver_destroy   cusolverDnDestroy

// =====> cusolver interface
#if   defined(PRECISION_z)
#define magma_zpotrf_batched_vendor           cusolverDnZpotrfBatched

#elif defined(PRECISION_c)
#define magma_cpotrf_batched_vendor           cusolverDnCpotrfBatched

#elif defined(PRECISION_d)
#define magma_dpotrf_batched_vendor           cusolverDnDpotrfBatched

#elif defined(PRECISION_s)
#define magma_spotrf_batched_vendor           cusolverDnSpotrfBatched

#else
#error "One of PRECISION_{s,d,c,z} must be defined."
#endif
#else
// =====> rocsolver interface
#define devsolver_handle_t  rocblas_handle
#define devsolver_create    rocblas_create_handle
#define devsolver_setstream rocblas_set_stream
#define devsolver_destroy   rocblas_destroy_handle

#if   defined(PRECISION_z)
#define magma_zpotrf_batched_vendor       rocsolver_zpotrf_batched
#define magma_zrocblas_scalar             rocblas_double_complex

#elif defined(PRECISION_c)
#define magma_cpotrf_batched_vendor       rocsolver_cpotrf_batched
#define magma_crocblas_scalar             rocblas_float_complex

#elif defined(PRECISION_d)
#define magma_dpotrf_batched_vendor       rocsolver_dpotrf_batched
#define magma_drocblas_scalar             double

#elif defined(PRECISION_s)
#define magma_spotrf_batched_vendor       rocsolver_spotrf_batched
#define magma_srocblas_scalar             float

#else
#error "One of PRECISION_{s,d,c,z} must be defined."
#endif
#endif

void
magma_zpotrf_batched_vendor(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    int *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    cublasFillMode_t uplo_ = (uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    magma_zpotrf_batched_vendor( handle, uplo_, (int)n,
                                 (cuDoubleComplex**)dA_array,  (int)ldda,
                                 (int*)dinfo_array, (int)batchCount );
    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zpotrf_batched_vendor( handle, uplo_, (int)n,
                                 (magma_zrocblas_scalar *const *)dA_array, (const int)ldda,
                                 (int*)dinfo_array, (int)batchCount );
    #endif
}

extern "C" magma_int_t
magma_zpotrf_lg_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue);

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
            magma_memset( dinfo_magma, 0, batchCount * sizeof(magma_int_t) );

            magma_zset_pointer( d_A_array, d_A, ldda, 0, 0, ldda * N, batchCount, queue );
            if(opts.version == 1)
            {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zpotrf_batched( opts.uplo, N, d_A_array, ldda, dinfo_magma, batchCount, queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if(opts.version == 2) {
                info = 0;
                devsolver_handle_t handle;
                devsolver_create(&handle);
                #ifdef MAGMA_HAVE_CUDA
                devsolver_setstream(handle, magma_queue_get_cuda_stream(opts.queue));
                #else
                devsolver_setstream(handle, magma_queue_get_hip_stream(opts.queue));
                #endif

                gpu_time = magma_sync_wtime( opts.queue );
                magma_zpotrf_batched_vendor(opts.uplo, N, d_A_array, ldda, dinfo_magma, batchCount, opts.queue, handle);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                devsolver_destroy(handle);
            }
            else {
                magma_int_t    nb = opts.nb;
                magma_int_t recnb = opts.nrhs;
                gpu_time = magma_sync_wtime( opts.queue );
                info = 0;
                magma_zpotrf_lg_batched(
                    opts.uplo, N, nb, recnb,
                    d_A_array, ldda, dinfo_magma,
                    batchCount, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
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
