/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c
       @author Chongxiao Cao
       @author Tingxing Dong
       @author Ahmad Abdelfattah
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

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zsyrk_vbatched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf=0., cpu_time=0.;
    double          error, magma_error, normalize, work[1];
    magma_int_t N, K, max_N, max_K;
    magma_int_t *An, *Ak;
    magma_int_t total_size_A_cpu = 0, total_size_C_cpu = 0;
    magma_int_t total_size_A_dev = 0, total_size_C_dev = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    magmaDoubleComplex *h_A, *h_C, *h_Cmagma;
    magmaDoubleComplex *d_A, *d_C;
    magma_int_t *h_N, *h_K, *d_N, *d_K;
    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magma_int_t *h_ldc, *h_lddc, *d_lddc;
    magmaDoubleComplex **h_A_array;
    magmaDoubleComplex **h_C_array;
    magmaDoubleComplex **d_A_array;
    magmaDoubleComplex **d_C_array;
    magmaDoubleComplex *h_A_tmp, *h_C_tmp, *h_Cmagma_tmp;
    
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(0.29, -0.48);
    magmaDoubleComplex beta  = MAGMA_Z_MAKE(-0.48, 0.29);
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;

    // allocate space for the sizes
    TESTING_CHECK( magma_imalloc_cpu(&h_N, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_K, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_lddc, batchCount) );

    TESTING_CHECK( magma_imalloc(&d_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_K, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_ldda, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_lddc, batchCount+1) );
    
    double *Anorm, *Cnorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Cnorm, batchCount ));
    
    // pointer arrays
    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_C_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_C_array, batchCount*sizeof(magmaDoubleComplex*)) );
    
    // See testing_zgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    #ifdef COMPLEX
    if (opts.transA == MagmaConjTrans) {
        opts.transA = MagmaTrans;
        printf("%% WARNING: transA = MagmaConjTrans changed to MagmaTrans\n");
    }
    #endif
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );

    printf("%%              max   max\n");
    printf("%% BatchCount     N     K   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            
            if ( opts.transA == MagmaNoTrans ) {
                h_lda = An = h_N;
                Ak = h_K;
            }
            else {
                h_lda = An = h_K;
                Ak = h_N;
            }
            h_ldc = h_N;
            
            // guarantee reproducible sizes
            srand(1000);
            
            gflops = 0;
            max_N = max_K = 0;
            
            total_size_A_cpu = total_size_C_cpu = 0;
            total_size_A_dev = total_size_C_dev = 0;
            
            for (int i = 0; i < batchCount; i++) {
                h_N[i] = 1 + (rand() % N);
                h_K[i] = 1 + (rand() % K);
                max_N = max( max_N, h_N[i]);
                max_K = max( max_K, h_K[i]);
                
                h_ldda[i] = magma_roundup( h_lda[i], opts.align );  // multiple of 32 by default
                h_lddc[i] = magma_roundup( h_ldc[i], opts.align );  // multiple of 32 by default
                
                total_size_A_cpu += Ak[i] * h_lda[i];
                total_size_C_cpu += h_N[i] * h_ldc[i];
                
                total_size_A_dev += Ak[i] * h_ldda[i];
                total_size_C_dev += h_N[i] * h_lddc[i];
                
                gflops += FLOPS_ZHERK( h_K[i], h_N[i] ) / 1e9;
            }
            
            TESTING_CHECK( magma_zmalloc_cpu(&h_A, total_size_A_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_C, total_size_C_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Cmagma, total_size_C_cpu) );
            
            TESTING_CHECK( magma_zmalloc(&d_A, total_size_A_dev) );
            TESTING_CHECK( magma_zmalloc(&d_C, total_size_C_dev)  );

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &total_size_A_cpu, h_A );
            lapackf77_zlarnv( &ione, ISEED, &total_size_C_cpu, h_C );
            
            // Compute norms for error
            h_A_tmp = h_A;
            h_C_tmp = h_C;
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_zlange( "F", &An[s], &Ak[s], h_A_tmp, &h_lda[s], work );
                Cnorm[s] = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &h_N[s], h_C_tmp, &h_ldc[s], work );
                h_A_tmp +=  Ak[s] * h_lda[s];
                h_C_tmp += h_N[s] * h_ldc[s];
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_K, 1, d_K, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_lddc, 1, d_lddc, 1, opts.queue );
            
            h_A_array[0] = d_A;
            h_C_array[0] = d_C;
            for (int i = 1; i < batchCount; i++) {
                h_A_array[i] = h_A_array[i-1] + Ak[i-1] * h_ldda[i-1];
                h_C_array[i] = h_C_array[i-1] + h_N[i-1] * h_lddc[i-1];
            }
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_A_array, 1, d_A_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_C_array, 1, d_C_array, 1, opts.queue );
            
            h_A_tmp = h_A;
            h_C_tmp = h_C;
            for (int i = 0; i < batchCount; i++) {
                magma_zsetmatrix( An[i], Ak[i], h_A_tmp, h_lda[i], h_A_array[i], h_ldda[i], opts.queue );
                magma_zsetmatrix( h_N[i], h_N[i], h_C_tmp, h_ldc[i], h_C_array[i], h_lddc[i], opts.queue );
                h_A_tmp += Ak[i] * h_lda[i];
                h_C_tmp += h_N[i] * h_ldc[i];
            }
            
            magma_time = magma_sync_wtime( opts.queue );
            #if 0
            magmablas_zsyrk_vbatched_max_nocheck(opts.uplo, opts.transA, d_N, d_K,
                             alpha, d_A_array, d_ldda,
                             beta,  d_C_array, d_lddc,
                             batchCount, max_N, max_K, opts.queue);
            #else
            magmablas_zsyrk_vbatched(opts.uplo, opts.transA, d_N, d_K,
                             alpha, d_A_array, d_ldda,
                             beta,  d_C_array, d_lddc,
                             batchCount, opts.queue);
            #endif
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            
            h_C_tmp = h_Cmagma;
            for (int i = 0; i < batchCount; i++) {
                magma_zgetmatrix( h_N[i], h_N[i], h_C_array[i], h_lddc[i], h_C_tmp, h_ldc[i], opts.queue );
                h_C_tmp += h_N[i] * h_ldc[i];
            }
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                // displace pointers for the cpu, reuse h_A_array, h_B_array, h_C_array
                h_A_array[0] = h_A;
                h_C_array[0] = h_C;
                for (int i = 1; i < batchCount; i++) {
                    h_A_array[i] = h_A_array[i-1] + Ak[i-1] * h_lda[i-1];
                    h_C_array[i] = h_C_array[i-1] + h_N[i-1] * h_ldc[i-1];
                }
                
                cpu_time = magma_wtime();
                //#define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                    blasf77_zsyrk( lapack_uplo_const(opts.uplo),
                                   lapack_trans_const(opts.transA),
                                   &h_N[i], &h_K[i],
                                   &alpha, h_A_array[i], &h_lda[i],
                                   &beta,  h_C_array[i], &h_ldc[i] );
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
                // compute error compared lapack
                // error = |dC - C| / (gamma_{k+2}|A||A| + gamma_2|Cin|)
                magma_error = 0;
                
                h_C_tmp = h_C;
                h_Cmagma_tmp = h_Cmagma;
                for (int s=0; s < batchCount; s++) {
                    normalize = sqrt(double(h_K[s]+2))*Anorm[s]*Anorm[s] + 2*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = h_ldc[s] * h_N[s];
                    blasf77_zaxpy( &Csize, &c_neg_one, h_C_tmp, &ione, h_Cmagma_tmp, &ione );
                    error = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &h_N[s], h_Cmagma_tmp, &h_ldc[s], work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                    
                    h_C_tmp      += h_N[s] * h_ldc[s];
                    h_Cmagma_tmp += h_N[s] * h_ldc[s];
                }
                
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                       (long long) batchCount, (long long) max_N, (long long) max_K,
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount, (long long) max_N, (long long) max_K,
                       magma_perf,  1000.*magma_time);
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_C  );
            magma_free_cpu( h_Cmagma  );

            magma_free( d_A );
            magma_free( d_C );
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    // free resources
    magma_free_cpu( h_N );
    magma_free_cpu( h_K );
    magma_free_cpu( h_ldda);
    magma_free_cpu( h_lddc );

    magma_free_cpu( Anorm );
    magma_free_cpu( Cnorm );

    magma_free_cpu( h_A_array  );
    magma_free_cpu( h_C_array  );
    
    magma_free( d_N );
    magma_free( d_K );
    magma_free( d_ldda );
    magma_free( d_lddc );
    magma_free( d_A_array );
    magma_free( d_C_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
