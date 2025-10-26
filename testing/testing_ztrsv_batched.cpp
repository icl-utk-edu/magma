/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Tingxing Dong

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


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time=0, device_perf=0, device_time=0, cpu_perf=0, cpu_time=0;
    double          magma_error, device_error, normx, normr, normA, work[1];
    magma_int_t i, j, s, N, info;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    magmaDoubleComplex *h_A, *h_b, *h_bdev, *h_bmagma, *h_blapack, *h_x;
    magmaDoubleComplex *d_A, *d_b;
    magmaDoubleComplex **d_A_array = NULL;
    magmaDoubleComplex **d_b_array = NULL;

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    int status = 0;
    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;

    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s, transA = %s, diag = %s \n",
           lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%% BatchCount   N   MAGMA Gflop/s (ms)   %s Gflop/s (ms)    CPU Gflop/s (ms)      MAGMA   %s error\n", g_platform_str, g_platform_str);
    printf("%%========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.msize[itest];

            gflops = FLOPS_ZTRSM(opts.side, N, 1) / 1e9 * batchCount;

            lda = Ak = N;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak*batchCount;
            sizeB = N*batchCount;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,       sizeA  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_b,       sizeB   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_x,       sizeB   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_blapack, sizeB   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_bdev, sizeB   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_bmagma,  sizeB   ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv,      Ak      ));

            TESTING_CHECK( magma_zmalloc( &d_A,       ldda*Ak*batchCount ));
            TESTING_CHECK( magma_zmalloc( &d_b,       N*batchCount  ));

            TESTING_CHECK( magma_malloc( (void**) &d_A_array,   batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_b_array,   batchCount * sizeof(magmaDoubleComplex*) ));
            memset( h_bmagma, 0, batchCount*N*sizeof(magmaDoubleComplex) );

            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            for (s=0; s < batchCount; s++) {
                magmaDoubleComplex* hAs = h_A + s * lda * Ak;
                lapackf77_zgetrf( &Ak, &Ak, hAs, &lda, ipiv, &info );
                for( j = 0; j < Ak; ++j ) {
                    for( i = 0; i < j; ++i ) {
                        hAs[j * lda + i] = MAGMA_Z_MUL( MAGMA_Z_MAKE(0.1, 0), hAs[i * lda + j] );
                    }
                }
            }

            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_b );
            memcpy( h_blapack, h_b, sizeB*sizeof(magmaDoubleComplex) );

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( Ak, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N,  batchCount, h_b, N, d_b, N, opts.queue );

            magma_zset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_zset_pointer( d_b_array, d_b, N, 0, 0, N, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ztrsv_batched(opts.uplo, opts.transA, opts.diag,
                                    N, d_A_array, ldda,
                                    d_b_array, 1, batchCount,
                                    opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_zgetmatrix( N, batchCount, d_b, N, h_bmagma, N, opts.queue );
            /* =====================================================================
               Performs operation using CUBLAS/HIPBLAS
               =================================================================== */
            magma_zsetmatrix( N, batchCount, h_b, N, d_b, N, opts.queue );
            magma_zset_pointer( d_b_array, d_b, N, 0, 0, N, batchCount, opts.queue );

            // CUBLAS version <= 6.0 has magmaDoubleComplex **            dA_array, no cast needed.
            // CUBLAS version    6.5 has magmaDoubleComplex const**       dA_array, requiring cast.
            // Correctly, it should be   magmaDoubleComplex const* const* dA_array, to avoid requiring cast.
            #if CUDA_VERSION >= 6050
                magmaDoubleComplex alpha = MAGMA_Z_ONE;
                device_time = magma_sync_wtime( opts.queue );
                cublasZtrsmBatched(
                    opts.handle, cublas_side_const(MagmaLeft), cublas_uplo_const(opts.uplo),
                    cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                    int(N), 1, &alpha,
                    (const magmaDoubleComplex**) d_A_array, int(ldda),
                    d_b_array, int(N), int(batchCount) );
                device_time = magma_sync_wtime( opts.queue ) - device_time;
                device_perf = gflops / device_time;
            #elif defined(MAGMA_HAVE_HIP)
                device_time = magma_sync_wtime( opts.queue );
                hipblasZtrsvBatched(
                    opts.handle, hipblas_uplo_const(opts.uplo), hipblas_trans_const(opts.transA), hipblas_diag_const(opts.diag),
                    (int)N, (const hipDoubleComplex **)d_A_array, (int)ldda,
                            (hipDoubleComplex **)d_b_array, (int)1, (int)batchCount );
                device_time = magma_sync_wtime( opts.queue ) - device_time;
                device_perf = gflops / device_time;
            #endif

            magma_zgetmatrix( N, batchCount, d_b, N, h_bdev, N, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (s=0; s < batchCount; s++) {
                    blasf77_ztrsv(
                        lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &N,
                        h_A + s * lda * Ak, &lda,
                        h_blapack + s * N,  &ione);
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - Ax|| / (||A||*||x||)
            magma_error  = 0;
            device_error = 0;
            for (s=0; s < batchCount; s++) {
                // error for CUBLAS/HIPBLAS
                normA = lapackf77_zlange( "F", &N, &N, h_A + s * lda * Ak, &lda, work );
                double err;

                #if CUDA_VERSION >= 6050 || defined(MAGMA_HAVE_HIP)
                normx = lapackf77_zlange( "F", &N, &ione, h_bdev + s * N, &ione, work );
                blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               h_A + s * lda * Ak, &lda,
                               h_bdev + s * N, &ione );

                blasf77_zaxpy( &N, &c_neg_one, h_b + s * N, &ione, h_bdev + s * N, &ione );
                normr = lapackf77_zlange( "F", &N, &ione, h_bdev + s * N, &N, work );
                err = normr / (normA*normx);

                if (std::isnan(err) || std::isinf(err)) {
                    printf("error for matrix %lld device_error = %7.2f where normr=%7.2f normx=%7.2f and normA=%7.2f\n",
                            (long long) s, err, normr, normx, normA);
                    device_error = err;
                    break;
                }
                device_error = max( err, device_error );
                #endif

                // error for MAGMA
                normx = lapackf77_zlange( "F", &N, &ione, h_bmagma + s * N, &ione, work );
                blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               h_A + s * lda * Ak, &lda,
                               h_bmagma  + s * N, &ione );

                blasf77_zaxpy( &N, &c_neg_one, h_b + s * N, &ione, h_bmagma + s * N, &ione );
                normr = lapackf77_zlange( "F", &N, &ione, h_bmagma + s * N, &N, work );
                err = normr / (normA*normx);

                if (std::isnan(err) || std::isinf(err)) {
                    printf("error for matrix %lld magma_error = %7.2f where normr=%7.2f normx=%7.2f and normA=%7.2f\n",
                            (long long) s, err, normr, normx, normA);
                    magma_error = err;
                    break;
                }
                magma_error = max( err, magma_error );
            }
            bool okay = (magma_error < tol && device_error < tol);
            status += ! okay;

            if ( opts.lapack ) {
                printf("%10lld %5lld    %7.2f (%7.2f)     %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                        (long long) batchCount, (long long) N,
                        magma_perf,  1000.*magma_time,
                        device_perf, 1000.*device_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, device_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                printf("%10lld %5lld    %7.2f (%7.2f)     %7.2f (%7.2f)     ---   (  ---  )   %8.2e   %8.2e   %s\n",
                        (long long) batchCount, (long long) N,
                        magma_perf,  1000.*magma_time,
                        device_perf, 1000.*device_time,
                        magma_error, device_error,
                        (okay ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_b );
            magma_free_cpu( h_x );
            magma_free_cpu( h_blapack );
            magma_free_cpu( h_bdev );
            magma_free_cpu( h_bmagma  );
            magma_free_cpu( ipiv );

            magma_free( d_A );
            magma_free( d_b );
            magma_free( d_A_array );
            magma_free( d_b_array );

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
