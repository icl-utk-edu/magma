/*
    -- MAGMA (version 1.1) --
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

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          magma_error, Cnorm, work[1];
    magma_int_t M, N;
    magma_int_t An;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t NN;
    magma_int_t batchCount;

    magmaDoubleComplex *h_A, *h_B, *h_C, *h_Cmagma;
    magmaDoubleComplex *d_A, *d_B, *d_C;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dB_array = NULL;
    magmaDoubleComplex **dC_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.check |= opts.lapack;
    batchCount = opts.batchcount;
    
    TESTING_CHECK( magma_malloc((void**)&dA_array, batchCount * sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&dB_array, batchCount * sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&dC_array, batchCount * sizeof(magmaDoubleComplex*)) );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    printf("%% If running lapack (option --lapack), MAGMA error is computed relative to CPU BLAS result.\n\n"
           "%% side = %s, uplo = %s\n",
           lapack_side_const(opts.side),
           lapack_uplo_const(opts.uplo));
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_ZHEMM(opts.side, M, N) / 1e9 * batchCount;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                An = M;
            } else {
                lda = N;
                An = N;
            }
            ldb = ldc = M;
            NN = N * batchCount;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An*batchCount;
            sizeB = ldb*N*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_CHECK( magma_zmalloc_cpu(&h_A, sizeA) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_B, sizeB) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_C, sizeC) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Cmagma, sizeC) );
            
            TESTING_CHECK( magma_zmalloc(&d_A, ldda*An*batchCount) );
            TESTING_CHECK( magma_zmalloc(&d_B, lddb*N*batchCount) );
            TESTING_CHECK( magma_zmalloc(&d_C, lddc*N*batchCount) );
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            
            /* Make A Hermitian */
            for(int i = 0; i < batchCount; i++){
                magma_zmake_hermitian( An, h_A + i*An*lda, lda );
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( An, An*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( M, N*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_zsetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_zset_pointer( dA_array, d_A, ldda, 0, 0, ldda*An, batchCount, opts.queue );
            magma_zset_pointer( dB_array, d_B, lddb, 0, 0, lddb*N, batchCount, opts.queue );
            magma_zset_pointer( dC_array, d_C, lddc, 0, 0, lddc*N, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_zhemm_batched( 
                    opts.side, opts.uplo, M, N, 
                    alpha, dA_array, ldda, 
                           dB_array, lddb, 
                    beta,  dC_array, lddc, 
                    batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_zgetmatrix( M, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );
            
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
                for (int i=0; i < batchCount; i++)
                {
                   blasf77_zhemm(
                               lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               &M, &N,
                               &alpha, h_A + i*lda*An, &lda,
                                       h_B + i*ldb*N, &ldb,
                               &beta,  h_C + i*ldc*N, &ldc );
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
            if ( opts.check ) {
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error  = 0;
                for (int s=0; s < batchCount; s++)
                {
                    magma_int_t csize = ldc * N;
 
                    Cnorm = lapackf77_zlange( "M", &M, &N, h_C + s*csize, &ldc, work );
                    blasf77_zaxpy( &csize, &c_neg_one, h_C + s*csize, &ione, h_Cmagma + s*csize, &ione );
                    double err = lapackf77_zlange( "M", &M, &N, h_Cmagma + s*csize, &ldc, work ) / Cnorm;
                    if ( isnan(err) || isinf(err) ) {
                        magma_error = err;
                        break;
                    }
                    magma_error = max( err, magma_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                   (int) batchCount, (int) M, (int) N,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10d %5d %5d   %7.2f (%7.2f)     ---   (  ---  )   ---\n",
                   (int) batchCount, (int) M, (int) N,
                   magma_perf,  1000.*magma_time );
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_C );
            magma_free_cpu( h_Cmagma );
            
            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_C );
            
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_free( dA_array );
    magma_free( dB_array );
    magma_free( dC_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
