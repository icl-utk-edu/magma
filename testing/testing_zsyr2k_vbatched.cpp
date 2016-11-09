/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c
       @author Mark Gates
       @author Azzam Haidar
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
   -- Testing zsyr2k_vbatched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          error, magma_error, normalize, work[1];
    magma_int_t N, K;
    magma_int_t *An, *Ak, *Bn, *Bk;
    magma_int_t total_size_A_cpu = 0, total_size_B_cpu = 0, total_size_C_cpu = 0;
    magma_int_t total_size_A_dev = 0, total_size_B_dev = 0, total_size_C_dev = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;
    magma_int_t max_N, max_K;

    magmaDoubleComplex *h_A, *h_B, *h_C, *h_Cmagma;
    magmaDoubleComplex *d_A, *d_B, *d_C;
    magmaDoubleComplex **h_A_array = NULL;
    magmaDoubleComplex **h_B_array = NULL;
    magmaDoubleComplex **h_C_array = NULL;
    magmaDoubleComplex **d_A_array = NULL;
    magmaDoubleComplex **d_B_array = NULL;
    magmaDoubleComplex **d_C_array = NULL;
    magmaDoubleComplex *h_A_tmp, *h_B_tmp, *h_C_tmp;
    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magma_int_t *h_ldb, *h_lddb, *d_lddb;
    magma_int_t *h_ldc, *h_lddc, *d_lddc;
    magma_int_t *h_N, *h_K; // hold the sizes on cpu
    magma_int_t *d_N, *d_K; // hold the sizes on gpu
    
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    
    // allocate space for the sizes/leading dim.
    TESTING_CHECK( magma_imalloc_cpu(&h_N, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_K, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_lddb, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_lddc, batchCount) );
    
    TESTING_CHECK( magma_imalloc(&d_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_K, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_ldda, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_lddb, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_lddc, batchCount+1) );
    
    double *Anorm, *Bnorm, *Cnorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Bnorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Cnorm, batchCount ));
    
    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_B_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_C_array, batchCount*sizeof(magmaDoubleComplex*)) );

    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_B_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_C_array, batchCount*sizeof(magmaDoubleComplex*)) );
    
    // See testing_zgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n"
           "%% transA = %s, transB = %s\n",
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB));
    
    #ifdef COMPLEX
    if (opts.transA == MagmaConjTrans) {
        opts.transA = MagmaTrans;
        printf("%% WARNING: transA = MagmaConjTrans changed to MagmaTrans\n");
    }
    #endif
    
    printf("%%              max   max\n");
    printf("%% BatchCount     N     K   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            
            if ( opts.transA == MagmaNoTrans ) {
                h_lda = An = h_N;
                h_ldb = Bn = h_N;
                Ak = h_K;
                Bk = h_K;
            }
            else {
                h_lda = An = h_K;
                h_ldb = Bn = h_K;
                Ak = h_N;
                Bk = h_N;
            }
            h_ldc = h_N;
            
            // guarantee reproducible sizes
            srand(1000);
            
            gflops = 0;
            max_N = max_K = 0;
            total_size_A_cpu = total_size_A_dev = 0;
            total_size_B_cpu = total_size_B_dev = 0;
            total_size_C_cpu = total_size_C_dev = 0;
            
            for (int i = 0; i < batchCount; i++) {
                h_N[i] = 1 + (rand() % N);
                h_K[i] = 1 + (rand() % K);
                max_N = max( max_N, h_N[i] );
                max_K = max( max_K, h_K[i] );
                
                h_ldda[i] = magma_roundup( h_lda[i], opts.align );  // multiple of 32 by default
                h_lddb[i] = magma_roundup( h_ldb[i], opts.align );  // multiple of 32 by default
                h_lddc[i] = magma_roundup( h_ldc[i], opts.align );  // multiple of 32 by default
                
                total_size_A_cpu += Ak[i] * h_lda[i];
                total_size_A_dev += Ak[i] * h_ldda[i];
                
                total_size_B_cpu += Bk[i] * h_ldb[i];
                total_size_B_dev += Bk[i] * h_lddb[i];
                
                total_size_C_cpu += h_N[i] * h_ldc[i];
                total_size_C_dev += h_N[i] * h_lddc[i];
            
                gflops += FLOPS_ZHER2K( h_N[i], h_K[i] ) / 1e9;
            }
            
            TESTING_CHECK( magma_zmalloc_cpu(&h_A, total_size_A_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_B, total_size_B_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_C, total_size_C_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Cmagma, total_size_C_cpu) );
            
            TESTING_CHECK( magma_zmalloc(&d_A, total_size_A_dev) );
            TESTING_CHECK( magma_zmalloc(&d_B, total_size_B_dev) );
            TESTING_CHECK( magma_zmalloc(&d_C, total_size_C_dev) );

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &total_size_A_cpu, h_A );
            lapackf77_zlarnv( &ione, ISEED, &total_size_B_cpu, h_B );
            lapackf77_zlarnv( &ione, ISEED, &total_size_C_cpu, h_C );
            
            // Compute norms for error
            h_A_tmp = h_A;
            h_B_tmp = h_B;
            h_C_tmp = h_C;
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_zlange( "F", &An[s], &Ak[s], h_A_tmp, &h_lda[s], work );
                Bnorm[s] = lapackf77_zlange( "F", &Bn[s], &Bk[s], h_B_tmp, &h_ldb[s], work );
                Cnorm[s] = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &h_N[s], h_C_tmp, &h_ldc[s], work );
                h_A_tmp +=  Ak[s] * h_lda[s];
                h_B_tmp +=  Bk[s] * h_ldb[s];
                h_C_tmp += h_N[s] * h_ldc[s];
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_K, 1, d_K, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_lddb, 1, d_lddb, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_lddc, 1, d_lddc, 1, opts.queue );
            
            h_A_array[0] = d_A;
            h_B_array[0] = d_B;
            h_C_array[0] = d_C;
            for (int i = 1; i < batchCount; i++) {
                h_A_array[i] = h_A_array[i-1] + Ak[i-1] * h_ldda[i-1];
                h_B_array[i] = h_B_array[i-1] + Bk[i-1] * h_lddb[i-1];
                h_C_array[i] = h_C_array[i-1] + h_N[i-1] * h_lddc[i-1];
            }
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_A_array, 1, d_A_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_B_array, 1, d_B_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_C_array, 1, d_C_array, 1, opts.queue );
            
            h_A_tmp = h_A;
            h_B_tmp = h_B;
            h_C_tmp = h_C;
            for (int i = 0; i < batchCount; i++) {
                magma_zsetmatrix( An[i], Ak[i], h_A_tmp, h_lda[i], h_A_array[i], h_ldda[i], opts.queue );
                magma_zsetmatrix( Bn[i], Bk[i], h_B_tmp, h_ldb[i], h_B_array[i], h_lddb[i], opts.queue );
                magma_zsetmatrix( h_N[i], h_N[i], h_C_tmp, h_ldc[i], h_C_array[i], h_lddc[i], opts.queue );
                h_A_tmp += Ak[i] * h_lda[i];
                h_B_tmp += Bk[i] * h_ldb[i];
                h_C_tmp += h_N[i] * h_ldc[i];
            }
            
            magma_time = magma_sync_wtime( opts.queue );
            #if 0
            magmablas_zsyr2k_vbatched_max_nocheck(opts.uplo, opts.transA, d_N, d_K,
                             alpha, d_A_array, d_ldda,
                                    d_B_array, d_lddb,
                             beta, d_C_array, d_lddc,
                             batchCount, max_N, max_K, opts.queue );
            #else
            magmablas_zsyr2k_vbatched(opts.uplo, opts.transA, d_N, d_K,
                             alpha, d_A_array, d_ldda,
                                    d_B_array, d_lddb,
                             beta, d_C_array, d_lddc,
                             batchCount, opts.queue );
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
                h_B_array[0] = h_B;
                h_C_array[0] = h_C;
                for (int i = 1; i < batchCount; i++) {
                    h_A_array[i] = h_A_array[i-1] + Ak[i-1] * h_lda[i-1];
                    h_B_array[i] = h_B_array[i-1] + Bk[i-1] * h_ldb[i-1];
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
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    blasf77_zsyr2k( lapack_uplo_const(opts.uplo),
                                    lapack_trans_const(opts.transA),
                                    &h_N[s], &h_K[s],
                                    &alpha, h_A_array[s], &h_lda[s],
                                            h_B_array[s], &h_ldb[s],
                                    &beta,  h_C_array[s], &h_ldc[s] );
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
                // error = |dC - C| / (2*gamma_{k+2}|A||B| + gamma_2|Cin|)
                magma_error = 0;

                h_C_tmp = h_C;
                magmaDoubleComplex* h_Cmagma_tmp = h_Cmagma;
                for (int s=0; s < batchCount; s++) {
                    normalize = 2*sqrt(double(h_K[s]+2))*Anorm[s]*Bnorm[s] + 2*Cnorm[s];
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

    // free resources
    magma_free_cpu( h_N );
    magma_free_cpu( h_K );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_lddb );
    magma_free_cpu( h_lddc );

    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );
    magma_free_cpu( Cnorm );

    magma_free_cpu( h_A_array );
    magma_free_cpu( h_B_array );
    magma_free_cpu( h_C_array );
    
    magma_free( d_N );
    magma_free( d_K );
    magma_free( d_ldda );
    magma_free( d_lddb );
    magma_free( d_lddc );
    magma_free( d_A_array );
    magma_free( d_B_array );
    magma_free( d_C_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
