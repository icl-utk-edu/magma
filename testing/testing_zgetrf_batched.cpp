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

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

// uncomment to introduce singularity in one matrix
// by setting two different columns to zeros
// (edit MTX_ID, COL1, and COL2 accordingly)
//#define SINGULARITY_CHECK
#ifdef SINGULARITY_CHECK
#define MTX_ID (10)    // checked against batchCount
#define COL1   (29)     // checked against #columns
#define COL2   (30)    // checked against #columns
#endif

////////////////////////////////////////////////////////////////////////////////
double get_LU_error(magma_int_t M, magma_int_t N,
                    magmaDoubleComplex *A,  magma_int_t lda,
                    magmaDoubleComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M, N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ZERO;
    magmaDoubleComplex *L, *U;
    double work[1], matnorm, residual;

    TESTING_CHECK( magma_zmalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( magma_zmalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaDoubleComplex) );

    lapackf77_zlaswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );

    matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);

    blasf77_zgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_Z_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);
}

////////////////////////////////////////////////////////////////////////////////
// compares the GPU output to the CPU's
void
get_LU_forward_error(
    magma_int_t M, magma_int_t N,
    magmaDoubleComplex *refLU, magma_int_t ref_lda, magma_int_t *refIPIV,
    magmaDoubleComplex *resLU, magma_int_t res_lda, magma_int_t *resIPIV,
    double* error, magma_int_t* pivots_match)
{
    magma_int_t i, j;
    double work[1], matnorm, residual;
    *error = MAGMA_D_ZERO;
    matnorm = lapackf77_zlange("f", &M, &N, refLU, &ref_lda, work);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            resLU[i+j*res_lda] = MAGMA_Z_SUB( resLU[i+j*res_lda], refLU[i+j*ref_lda] );
        }
    }
    residual = lapackf77_zlange("f", &M, &N, resLU, &res_lda, work);
    *error = residual / (matnorm * N);

    // compare pivots
    *pivots_match = 1;
    for( i = 0; i < min(M,N); i++ ) {
        if( !( refIPIV[i] == resIPIV[i]) ) {
            *pivots_match = 0;
            break;
        }
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, device_perf=0, device_time=0, cpu_perf=0, cpu_time=0;
    double          error;
    magmaDoubleComplex *h_A, *h_R, *h_Amagma;
    magmaDoubleComplex *dA;
    magmaDoubleComplex **dA_array = NULL;

    magma_int_t     **dipiv_array = NULL;
    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *dipiv_magma, *dinfo_magma;
    int             *dipiv_device, *dinfo_device;  // not magma_int_t
    #ifdef MAGMA_HAVE_SYCL
    std::int64_t *dipiv_syclblas;
    #endif
    magma_int_t M, N, n2, lda, ldda, min_mn, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= (opts.check == 2);
    double tol   = opts.tolerance * lapackf77_dlamch("E");
    batchCount   = opts.batchcount;
    magma_int_t columns;

    if(opts.check == 2) {
        printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   %s Gflop/s (ms)   ||Aref-A||/(||Aref||*N)\n", g_platform_str);
    }
    else{
        printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   %s Gflop/s (ms)   ||PA-LU||/(||A||*N)\n", g_platform_str);
    }
    printf("%%==========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZGETRF( M, N ) / 1e9 * batchCount;

            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn * batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,  n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Amagma,  n2     ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,  n2     ));

            TESTING_CHECK( magma_zmalloc( &dA,  ldda*N * batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv_magma,  min_mn * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_device, min_mn * batchCount * sizeof(int) ));  // not magma_int_t
            TESTING_CHECK( magma_malloc( (void**) &dinfo_device, batchCount          * sizeof(int) ));
            #ifdef MAGMA_HAVE_SYCL
	    TESTING_CHECK( magma_malloc( (void**) &dipiv_syclblas, min_mn * batchCount * sizeof(std::int64_t) ));  // not magma_int_t
            #endif													

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );

            #ifdef SINGULARITY_CHECK
            // introduce singularity -- for debugging purpose only
            magma_int_t id   = min(MTX_ID, batchCount-1);
            magma_int_t col1 = min(COL1, N-1);
            magma_int_t col2 = min(COL2, N-1);
            printf("singularity in matrix %lld of size (%lld, %lld) : col. %lld & %lld set to zeros\n",
                   (long long)id, (long long)M, (long long)N,
                   (long long)col1, (long long)col2);
            memset(h_A + id*lda*N + col1 * lda, 0, M * sizeof(magmaDoubleComplex));
            memset(h_A + id*lda*N + col2 * lda, 0, M * sizeof(magmaDoubleComplex));
            #endif

            columns = N * batchCount;
            lapackf77_zlacpy( MagmaFullStr, &M, &columns, h_A, &lda, h_R, &lda );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, columns, h_R, lda, dA, ldda, opts.queue );
            magma_zset_pointer( dA_array, dA, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_iset_pointer( dipiv_array, dipiv_magma, 1, 0, 0, min_mn, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            info = magma_zgetrf_batched( M, N, dA_array, ldda, dipiv_array,  dinfo_magma, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_zgetmatrix( M, N*batchCount, dA, ldda, h_Amagma, lda, opts.queue );

            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );

            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgetrf_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                }
            }

            if (info != 0) {
                printf("magma_zgetrf_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            /* ====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( M, columns, h_R, lda, dA,  ldda, opts.queue );
            magma_zset_pointer( dA_array, dA, ldda, 0, 0, ldda * N, batchCount, opts.queue );

            device_time = magma_sync_wtime( opts.queue );
            if (M == N ) {
                #ifdef MAGMA_HAVE_CUDA
                cublasZgetrfBatched( opts.handle, int(N),
                                     dA_array, int(ldda), dipiv_device,
                                     dinfo_device, int(batchCount) );
                #elif defined(MAGMA_HAVE_HIP)
                hipblasZgetrfBatched( opts.handle, int(N),
                                     (hipblasDoubleComplex**)dA_array, int(ldda), dipiv_device,
                                     dinfo_device, int(batchCount) );
                #elif defined(MAGMA_HAVE_SYCL)
		std::int64_t scratchSize = oneapi::mkl::lapack::getrf_batch_scratchpad_size<magmaDoubleComplex>(
				*opts.handle, (std::int64_t*)(&N), (std::int64_t*)(&N), (std::int64_t*)(&ldda),
				 std::int64_t(1), (std::int64_t*)(&batchCount));
		magmaDoubleComplex *scratchPad;
                TESTING_CHECK( magma_zmalloc( &scratchPad,  scratchSize));
		oneapi::mkl::lapack::getrf_batch( *opts.handle, (std::int64_t*)(&N), (std::int64_t*)(&N),
				(magmaDoubleComplex**)dA_array, (std::int64_t*)(&ldda), &dipiv_syclblas,
				std::int64_t(1), (std::int64_t*)(&batchCount),
			        scratchPad, scratchSize, {});
                #endif
            }
            else {
                printf("M != N, %s required M == N; %s is disabled\n", g_platform_str, g_platform_str);
            }
            device_time = magma_sync_wtime( opts.queue ) - device_time;
            device_perf = gflops / device_time;

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
                    lapackf77_zgetrf(&M, &N, h_A + s * lda * N, &lda, ipiv + s * min_mn, &locinfo);
                    if (locinfo != 0) {
                        printf("lapackf77_zgetrf matrix %lld returned error %lld: %s.\n",
                               (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)     %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000.,
                       device_perf, device_time*1000.  );
            }
            else {
                printf("%10lld %5lld %5lld     ---   (  ---  )    %7.2f (%7.2f)     %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf, magma_time*1000.,
                       device_perf, device_time*1000. );
            }

            if ( opts.check == 1 ) {
                magma_getvector( min_mn * batchCount, sizeof(magma_int_t), dipiv_magma, 1, ipiv, 1, opts.queue );
                error = 0;
                for (int i=0; i < batchCount; i++) {
                    for (int k=0; k < min_mn; k++) {
                        if (ipiv[i*min_mn+k] < 1 || ipiv[i*min_mn+k] > M ) {
                            printf("error for matrix %lld ipiv @ %lld = %lld\n",
                                    (long long) i, (long long) k, (long long) ipiv[i*min_mn+k] );
                            error = -1;
                        }
                    }
                    if (error == -1) {
                        break;
                    }

                    double err = get_LU_error( M, N, h_R + i * lda*N, lda, h_Amagma + i * lda*N, ipiv + i * min_mn);
                    if (std::isnan(err) || std::isinf(err)) {
                        error = err;
                        break;
                    }
                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else if( opts.check == 2 ) {
                magma_int_t* ipiv_magma = NULL;
                TESTING_CHECK( magma_imalloc_cpu( &ipiv_magma,     batchCount * min_mn ));
                magma_getvector( batchCount * min_mn, sizeof(magma_int_t), dipiv_magma, 1, ipiv_magma, 1, opts.queue );

                double err = 0; error = 0;
                magma_int_t pivots_match = 1, piv_match = 1;
                for(magma_int_t s = 0; s < batchCount; s++) {
                    get_LU_forward_error(
                        M, N,
                        h_A      + s * lda * N, lda, ipiv       + s * min_mn,
                        h_Amagma + s * lda * N, lda, ipiv_magma + s * min_mn,
                        &err, &piv_match);

                    error = max(error, err);
                    pivots_match &= piv_match;
                }

                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e   %15s   %s\n", error,
                       (pivots_match == 1)  ? "pivots match" : "pivots mismatch",
                       (okay ? "ok" : "failed") );

                magma_free_cpu( ipiv_magma );
            }
            else {
                printf("     ---\n");
            }

            magma_free_cpu( cpu_info );
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            magma_free_cpu( h_Amagma );
            magma_free_pinned( h_R );

            magma_free( dA );
            magma_free( dinfo_magma );
            magma_free( dipiv_magma );
            magma_free( dipiv_device );
            magma_free( dinfo_device );
            magma_free( dipiv_syclblas );
            magma_free( dipiv_array );
            magma_free( dA_array );
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
