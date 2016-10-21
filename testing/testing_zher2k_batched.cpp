/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
       @author Azzam Haidar
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
#endif
#include "../control/magma_threadsetting.h"  // internal header

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zher2k_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          magma_error, Cnorm, work[1];
    magma_int_t N, K;
    magma_int_t An, Ak, Bn, Bk;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaDoubleComplex *h_A, *h_B, *h_C, *h_Cmagma;
    magmaDoubleComplex *d_A, *d_B, *d_C;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    
    magmaDoubleComplex **A_array = NULL;
    magmaDoubleComplex **B_array = NULL;
    magmaDoubleComplex **C_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    
    batchCount = opts.batchcount;

    double tol = opts.tolerance * lapackf77_dlamch("E");
    opts.lapack |= opts.check;
    
    TESTING_CHECK( magma_malloc((void**)&A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&B_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&C_array, batchCount*sizeof(magmaDoubleComplex*)) );
            
    printf("%% If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "%% uplo = %s, trans = %s\n",
           lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA));
    #if defined (PRECISION_c) || defined(PRECISION_z)
    if(opts.transA == MagmaTrans){
        opts.transA = MagmaConjTrans;
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #else
    if(opts.transA == MagmaConjTrans){
        opts.transA = MagmaTrans;
        printf("%% WARNING: transA = MagmaConjTrans changed to MagmaTrans\n");
    }
    #endif
    
    printf("%% BatchCount   N     K   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)    MAGMA error\n");
    printf("%%======================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = (batchCount * FLOPS_ZHER2K( K, N )) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                ldb = Bn = N;
                Ak = K;
                Bk = K;
            } else {
                lda = An = K;
                ldb = Bn = K;
                Ak = N;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak*batchCount;
            sizeB = ldb*Bk*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_CHECK( magma_zmalloc_cpu(&h_A, sizeA) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_B, sizeB) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_C, sizeC) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Cmagma, sizeC) );

            TESTING_CHECK( magma_zmalloc(&d_A, ldda*Ak*batchCount) );
            TESTING_CHECK( magma_zmalloc(&d_B, lddb*Bk*batchCount) );
            TESTING_CHECK( magma_zmalloc(&d_C, lddc*N*batchCount)  );

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            for (int i=0; i < batchCount; i++){
                magma_zmake_hermitian( N, h_C + i * ldc * N, ldc );
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( An, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( Bn, Bk*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_zsetmatrix( N, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_zset_pointer( A_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_zset_pointer( B_array, d_B, lddb, 0, 0, lddb*Bk, batchCount, opts.queue );
            magma_zset_pointer( C_array, d_C, lddc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_zher2k_batched(opts.uplo, opts.transA, N, K, 
                             alpha, A_array, ldda, 
                                    B_array, lddb, 
                             beta, C_array, lddc, batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_zgetmatrix( N, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );
            
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
                    blasf77_zher2k( lapack_uplo_const(opts.uplo),
                                    lapack_trans_const(opts.transA),
                                    &N, &K,
                                    &alpha, h_A + i*lda*Ak, &lda,
                                            h_B + i*ldb*Bk, &ldb,
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
            if ( opts.lapack ) {
                // compute relative error of magma relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error  = 0;
                for (int s=0; s < batchCount; s++)
                {
                    magma_int_t C_batchSize = ldc * N;
                    
                    Cnorm = safe_lapackf77_zlanhe( "fro", lapack_uplo_const(opts.uplo), &N, h_C + s*C_batchSize, &ldc, work );

                    // ----- magma error
                    blasf77_zaxpy( &C_batchSize, &c_neg_one, h_C + s*C_batchSize, &ione, h_Cmagma + s*C_batchSize, &ione );
                    double err = safe_lapackf77_zlanhe( "fro", lapack_uplo_const(opts.uplo), &N, h_Cmagma + s*C_batchSize, &ldc, work ) / Cnorm;

                    if ( isnan(err) || isinf(err) ) {
                        magma_error = err;
                        break;
                    }
                    magma_error = max( err, magma_error );
                }
                
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%10lld %5lld %5lld    %7.2f (%7.2f)     %7.2f (%7.2f)     %8.2e  %s\n",
                   (long long) batchCount, (long long) N, (long long) K,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (okay ? "ok" : "failed"));
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_C  );
            magma_free_cpu( h_Cmagma  );

            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_C );
            
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_free( A_array );
    magma_free( B_array );
    magma_free( C_array );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
